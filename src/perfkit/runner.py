"""压测执行器。

两种负载模式，取舍见 docs/methodology.md 的 Load models 一节：
  closed：固定并发闭环，一个请求完成立刻补下一个，用于并发阶梯找饱和点
  open：泊松到达率开环，到达不受服务端快慢影响，用于测真实排队行为

闭环模式测不出排队崩溃——服务变慢时到达率自动降低，形成隐式背压。
真实生产是开环的，所以最终限流值必须用 open 模式的结果。
"""
from __future__ import annotations

import asyncio
import contextlib
import random
import time
from collections.abc import Callable
from typing import Any

import httpx

from .client import make_async_client, stream_request
from .config import ScenarioSpec, SLOConfig, TargetConfig
from .metrics import MetricsCollector
from .record import JsonlWriter, RequestRecord
from .stats import ScenarioStats, aggregate
from .tokens import PromptFactory, TokenCounter, make_messages

ProgressFn = Callable[[str], None]


class ScenarioRunner:
    def __init__(
        self,
        target: TargetConfig,
        counter: TokenCounter,
        collector: MetricsCollector | None = None,
        writer: JsonlWriter | None = None,
        progress: ProgressFn | None = None,
        seed: int = 0,
    ):
        self.target = target
        self.counter = counter
        self.collector = collector
        self.writer = writer
        self.progress = progress or (lambda _m: None)
        self.seed = seed

    def _log(self, msg: str) -> None:
        self.progress(msg)

    async def _one(
        self,
        client: httpx.AsyncClient,
        spec: ScenarioSpec,
        factory: PromptFactory,
        idx: int,
        is_warmup: bool,
    ) -> RequestRecord | None:
        prompt, reuse = factory.build(
            idx, spec.input_tokens, spec.prefix_mode, spec.shared_prefix_ratio
        )
        n_in = self.counter.count(prompt)
        rec = await stream_request(
            client=client,
            target=self.target,
            messages=make_messages(prompt),
            max_tokens=spec.output_tokens,
            temperature=spec.temperature,
            ignore_eos=spec.ignore_eos,
            workload_class=spec.name,
            input_tokens=n_in,
            target_output_tokens=spec.output_tokens,
            concurrency=spec.parallelism,
            expect_prefix_reuse=reuse,
        )
        # 预热请求不进统计：首批请求含 kernel 编译与 cache 冷启动
        if is_warmup:
            return None
        if self.writer:
            self.writer.write(rec)
        return rec

    async def _run_closed(
        self, client: httpx.AsyncClient, spec: ScenarioSpec, factory: PromptFactory
    ) -> list[RequestRecord]:
        """固定并发：N 个 worker 各自串行取任务，总在飞请求恒为 N。"""
        total = spec.num_requests
        deadline = time.perf_counter() + spec.duration_s if spec.duration_s else None
        results: list[RequestRecord] = []
        counter = {"issued": 0, "done": 0}
        lock = asyncio.Lock()

        async def worker() -> None:
            while True:
                async with lock:
                    if total is not None and counter["issued"] >= total:
                        return
                    if deadline is not None and time.perf_counter() >= deadline:
                        return
                    idx = counter["issued"]
                    counter["issued"] += 1
                rec = await self._one(client, spec, factory, idx, is_warmup=False)
                if rec is not None:
                    results.append(rec)
                async with lock:
                    counter["done"] += 1
                    d = counter["done"]
                if total and d % max(1, total // 10) == 0:
                    self._log(f"    {spec.name} 进度 {d}/{total}")

        await asyncio.gather(*[worker() for _ in range(spec.parallelism)])
        return results

    async def _run_open(
        self, client: httpx.AsyncClient, spec: ScenarioSpec, factory: PromptFactory
    ) -> list[RequestRecord]:
        """泊松到达：间隔取指数分布，不等前一个请求完成。

        在飞请求数不设上限——服务扛不住时会堆积，这正是要观测的现象。
        """
        rng = random.Random(self.seed + 7)
        rate = spec.request_rate or 1.0
        total = spec.num_requests
        deadline = time.perf_counter() + spec.duration_s if spec.duration_s else None
        tasks: list[asyncio.Task] = []
        idx = 0

        while True:
            if total is not None and idx >= total:
                break
            if deadline is not None and time.perf_counter() >= deadline:
                break
            tasks.append(
                asyncio.create_task(
                    self._one(client, spec, factory, idx, is_warmup=False)
                )
            )
            idx += 1
            await asyncio.sleep(rng.expovariate(rate))

        self._log(f"    {spec.name} 已发出 {idx} 个请求，等待收敛")
        done = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in done if isinstance(r, RequestRecord)]

    async def run(self, spec: ScenarioSpec, slo: SLOConfig) -> ScenarioStats:
        eff_slo = slo.per_scenario(spec.slo)
        factory = PromptFactory(self.counter, seed=self.seed + hash(spec.name) % 10000)

        pool = spec.parallelism if spec.mode == "closed" else max(
            32, int((spec.request_rate or 1) * 8)
        )
        client = make_async_client(self.target, pool)

        try:
            if spec.warmup_requests > 0:
                self._log(f"    预热 {spec.warmup_requests} 个请求（不计入统计）")
                await asyncio.gather(
                    *[
                        self._one(client, spec, factory, -(i + 1), is_warmup=True)
                        for i in range(spec.warmup_requests)
                    ]
                )

            before = self.collector.snapshot() if self.collector else None
            sampler = None
            if self.collector and spec.mode:
                sampler = asyncio.create_task(self._sample_loop())

            t0 = time.perf_counter()
            if spec.mode == "closed":
                recs = await self._run_closed(client, spec, factory)
            else:
                recs = await self._run_open(client, spec, factory)
            elapsed = time.perf_counter() - t0

            if sampler:
                sampler.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await sampler

            after = self.collector.snapshot() if self.collector else None
            server = (
                self.collector.diff(before, after)
                if self.collector and before and after
                else None
            )
        finally:
            await client.aclose()

        self._log(f"    {spec.name} 完成，{len(recs)} 条记录，耗时 {elapsed:.1f}s")

        return aggregate(
            name=spec.name,
            mode=spec.mode,
            parallelism=spec.parallelism,
            request_rate=spec.request_rate,
            target_in=spec.input_tokens,
            target_out=spec.output_tokens,
            records=recs,
            slo_ttft_ms=eff_slo.ttft_ms,
            slo_tpot_ms=eff_slo.tpot_ms,
            slo_e2e_ms=eff_slo.e2e_ms,
            stall_itl_ms=eff_slo.stall_itl_ms,
            server_metrics=server,
            request_multiplier=spec.request_multiplier,
            prefix_mode=spec.prefix_mode,
            temperature=spec.temperature,
            stream=spec.stream,
            api_url=self.target.chat_url,
            model=self.target.model,
            warmup_requests=spec.warmup_requests,
            ignore_eos=spec.ignore_eos,
            duration_cap_s=spec.duration_s,
        )

    async def _sample_loop(self, interval: float = 1.0) -> None:
        """周期采 gauge，抓 waiting / KV 使用率峰值。"""
        while True:
            await asyncio.sleep(interval)
            # 采样失败不影响压测，静默跳过这一拍
            with contextlib.suppress(Exception):
                self.collector.sample_gauges()  # type: ignore[union-attr]


async def probe_target(target: TargetConfig) -> dict[str, Any]:
    """探测能力：/v1/models、/metrics、/tokenize。

    压测前必须跑，否则 has_metrics / has_tokenize 为 False 时会静默退化为
    估算值，报告里看不出来。
    """
    info: dict[str, Any] = {"reachable": False}
    async with httpx.AsyncClient(timeout=15.0) as c:
        try:
            r = await c.get(target.models_url, headers=target.headers())
            r.raise_for_status()
            data = r.json().get("data", [])
            info["reachable"] = True
            info["models"] = [m.get("id") for m in data]
            for m in data:
                if m.get("id") == target.model:
                    target.max_model_len = m.get("max_model_len")
                    info["max_model_len"] = target.max_model_len
                    break
            if target.model not in info["models"]:
                info["model_mismatch"] = (
                    f"配置的 model={target.model} 不在服务列表 {info['models']} 中"
                )
        except Exception as exc:  # noqa: BLE001
            info["error"] = str(exc)[:300]
            return info

        try:
            r = await c.get(target.metrics_url, headers=target.headers())
            target.has_metrics = r.status_code == 200
        except Exception:  # noqa: BLE001
            target.has_metrics = False
        info["has_metrics"] = target.has_metrics

        try:
            r = await c.post(
                target.tokenize_url,
                headers=target.headers(),
                json={"model": target.model, "prompt": "probe"},
            )
            target.has_tokenize = r.status_code == 200
        except Exception:  # noqa: BLE001
            target.has_tokenize = False
        info["has_tokenize"] = target.has_tokenize

    return info
