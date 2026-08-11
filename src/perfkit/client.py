"""流式请求客户端。

不用 openai SDK 而直接用 httpx 手工解析 SSE，原因是 SDK 内部有缓冲和对象
构造开销，会污染 TTFT 与 ITL 的计时。这里在 socket 收到 chunk 的第一时间
就打时间戳。
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from datetime import datetime, timezone

import httpx

from .config import TargetConfig
from .record import RequestRecord


def _classify_error(exc: Exception) -> str:
    """错误分类按类型统计可靠性指标，不能只报一个总失败数。"""
    if isinstance(exc, asyncio.TimeoutError):
        return "timeout"
    if isinstance(exc, httpx.ConnectTimeout):
        return "connect_timeout"
    if isinstance(exc, httpx.ReadTimeout):
        return "read_timeout"
    if isinstance(exc, httpx.ConnectError):
        return "connect_error"
    if isinstance(exc, httpx.RemoteProtocolError):
        return "protocol_error"
    if isinstance(exc, httpx.HTTPStatusError):
        return f"http_{exc.response.status_code}"
    return type(exc).__name__


async def stream_request(
    client: httpx.AsyncClient,
    target: TargetConfig,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    ignore_eos: bool,
    workload_class: str,
    input_tokens: int,
    target_output_tokens: int,
    concurrency: int,
    expect_prefix_reuse: bool = False,
) -> RequestRecord:
    """发一个流式请求并采全部延迟指标。

    异常不向上抛：单个请求失败不应中断整轮压测，失败信息记进 record
    由 stats 汇总成 error_breakdown。
    """
    rec = RequestRecord(
        request_id=str(uuid.uuid4()),
        model=target.model,
        workload_class=workload_class,
        input_tokens=input_tokens,
        target_input_tokens=input_tokens,
        target_output_tokens=target_output_tokens,
        concurrency=concurrency,
    )

    payload: dict = {
        "model": target.model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        # 让服务端在最后一个 chunk 带上 usage，拿到权威 token 数
        "stream_options": {"include_usage": True},
    }
    if ignore_eos:
        # vLLM 扩展字段：强制生成到 max_tokens，保证各请求输出定长可比
        payload["ignore_eos"] = True

    rec.send_wall = datetime.now(timezone.utc).isoformat()
    rec.send_time = time.perf_counter()

    try:
        async with client.stream(
            "POST", target.chat_url, headers=target.headers(), json=payload
        ) as resp:
            rec.http_status = resp.status_code
            if resp.status_code != 200:
                body = await resp.aread()
                rec.status = "error"
                rec.error_type = f"http_{resp.status_code}"
                rec.error_detail = body.decode("utf-8", "replace")[:500]
                rec.end_time = time.perf_counter()
                return rec

            rec.instance = resp.headers.get("x-vllm-instance") or resp.headers.get(
                "server"
            )

            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break

                now = time.perf_counter()
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                # 末尾 usage-only chunk 的 choices 为空，不能当作 token
                usage = chunk.get("usage")
                if usage:
                    rec.usage_prompt_tokens = usage.get("prompt_tokens")
                    rec.usage_completion_tokens = usage.get("completion_tokens")

                choices = chunk.get("choices") or []
                if not choices:
                    continue

                ch = choices[0]
                delta = ch.get("delta") or {}
                content = delta.get("content")

                if ch.get("finish_reason"):
                    rec.finish_reason = ch["finish_reason"]

                # 空 content 的 role chunk 不计为 token，否则 TTFT 偏小
                if content:
                    if rec.first_token_time is None:
                        rec.first_token_time = now
                    rec.token_times.append(now)

            rec.end_time = time.perf_counter()

            # 输出 token 数以服务端 usage 为准，缺失时退化为 chunk 计数。
            # chunk 数与 token 数并不严格相等（一个 chunk 可能含多个 token），
            # 所以 usage 可用时必须优先。
            if rec.usage_completion_tokens is not None:
                rec.output_tokens = rec.usage_completion_tokens
            else:
                rec.output_tokens = len(rec.token_times)
            if rec.usage_prompt_tokens is not None:
                rec.input_tokens = rec.usage_prompt_tokens

            if rec.first_token_time is None:
                # 连上了但一个 token 都没吐：空响应，属于部署正确性问题，不是性能问题
                rec.status = "error"
                rec.error_type = "empty_response"
            else:
                rec.status = "ok"

    except Exception as exc:  # noqa: BLE001 - 需要吞掉所有异常保证压测继续
        rec.end_time = time.perf_counter()
        rec.status = "error"
        rec.error_type = _classify_error(exc)
        rec.error_detail = str(exc)[:500]

    return rec


def make_async_client(target: TargetConfig, concurrency: int) -> httpx.AsyncClient:
    """连接池上限必须 >= 并发数，否则请求会在客户端排队，

    表现为 TTFT 虚高，而服务端 queue_time 却正常，是个很难定位的坑。
    """
    limits = httpx.Limits(
        max_connections=concurrency + 16,
        max_keepalive_connections=concurrency + 16,
    )
    timeout = httpx.Timeout(
        connect=target.connect_timeout_s,
        read=target.read_timeout_s,
        write=30.0,
        pool=60.0,
    )
    return httpx.AsyncClient(limits=limits, timeout=timeout, http2=False)
