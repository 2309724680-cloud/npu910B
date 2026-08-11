"""采集 vLLM /metrics，作为客户端测量的交叉校验。

客户端算出来的 TTFT 含网络往返，服务端 histogram 不含。两者差值即网络与
客户端开销，超过 10% 说明测试机或链路是瓶颈，此时的数据不代表服务能力。
两个来源必须分别报告，不能混用。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import TargetConfig

# 计数器：取压测前后差值
_COUNTERS = (
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:request_success_total",
    "vllm:num_preemptions_total",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
)

# 瞬时量：取采样期间的最大与末值
_GAUGES = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
    "vllm:gpu_cache_usage_perc",
)

# histogram：取压测前后 bucket 差值再估分位数
_HISTOGRAMS = (
    "vllm:time_to_first_token_seconds",
    "vllm:request_time_per_output_token_seconds",
    "vllm:inter_token_latency_seconds",
    "vllm:e2e_request_latency_seconds",
    "vllm:request_queue_time_seconds",
    "vllm:request_prefill_time_seconds",
    "vllm:request_decode_time_seconds",
)

_LINE = re.compile(r"^(?P<name>[a-zA-Z_:][\w:]*)(?P<labels>\{[^}]*\})?\s+(?P<val>[^\s]+)$")


def _parse(text: str) -> dict[str, list[tuple[dict[str, str], float]]]:
    """解析 Prometheus 文本格式。多 label 组合各自保留。"""
    out: dict[str, list[tuple[dict[str, str], float]]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        try:
            val = float(m.group("val"))
        except ValueError:
            continue
        labels: dict[str, str] = {}
        lb = m.group("labels")
        if lb:
            for part in lb[1:-1].split(","):
                if "=" not in part:
                    continue
                k, _, v = part.partition("=")
                labels[k.strip()] = v.strip().strip('"')
        out.setdefault(m.group("name"), []).append((labels, val))
    return out


def _sum(parsed: dict, name: str) -> float | None:
    items = parsed.get(name)
    if not items:
        return None
    return sum(v for _, v in items)


def _buckets(parsed: dict, hist: str) -> list[tuple[float, float]]:
    """取 histogram 的 (le, cumulative_count)，按 le 升序。"""
    items = parsed.get(f"{hist}_bucket")
    if not items:
        return []
    acc: dict[float, float] = {}
    for labels, val in items:
        le = labels.get("le")
        if le is None:
            continue
        bound = float("inf") if le in ("+Inf", "Inf") else float(le)
        acc[bound] = acc.get(bound, 0.0) + val
    return sorted(acc.items())


def _hist_pct(buckets: list[tuple[float, float]],
              pct: float) -> tuple[float, bool] | None:
    """从累积 bucket 估分位数，返回 (值, 是否仅为上界)。

    vLLM 的 queue_time / prefill_time 最低分桶是 le=0.3s，粒度远大于实际值。
    当目标分位数落在首个分桶内时，桶内分布未知，线性插值会凭空造出一个
    数量级错误的结果（实测：真实排队约 0ms，插值报 150ms）。这种情况只返回
    桶上界并标记 upper_bound_only，由报告层显式标注为 "<= X"。
    """
    if not buckets:
        return None
    total = buckets[-1][1]
    if total <= 0:
        return None
    target = total * pct / 100.0
    prev_bound, prev_count = 0.0, 0.0
    for i, (bound, count) in enumerate(buckets):
        if count >= target:
            if bound == float("inf"):
                return (prev_bound, True)
            # 首桶命中：桶内无分布信息，只能给上界
            if i == 0:
                return (bound, True)
            span = count - prev_count
            if span <= 0:
                return (bound, True)
            frac = (target - prev_count) / span
            return (prev_bound + (bound - prev_bound) * frac, False)
        prev_bound, prev_count = bound, count
    return None


@dataclass
class MetricsSnapshot:
    ok: bool
    raw: dict[str, Any] = field(default_factory=dict)
    counters: dict[str, float] = field(default_factory=dict)
    gauges: dict[str, float] = field(default_factory=dict)
    histograms: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    error: str | None = None


class MetricsCollector:
    def __init__(self, target: TargetConfig):
        self.target = target
        self._client = httpx.Client(timeout=15.0)
        self._peak_gauges: dict[str, float] = {}

    def snapshot(self) -> MetricsSnapshot:
        if not self.target.has_metrics:
            return MetricsSnapshot(ok=False, error="metrics_disabled")
        try:
            r = self._client.get(self.target.metrics_url,
                                 headers=self.target.headers())
            r.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            return MetricsSnapshot(ok=False, error=str(exc)[:200])

        parsed = _parse(r.text)
        counters = {n: v for n in _COUNTERS if (v := _sum(parsed, n)) is not None}
        gauges = {n: v for n in _GAUGES if (v := _sum(parsed, n)) is not None}
        for k, v in gauges.items():
            self._peak_gauges[k] = max(self._peak_gauges.get(k, 0.0), v)
        hists = {h: b for h in _HISTOGRAMS if (b := _buckets(parsed, h))}
        return MetricsSnapshot(ok=True, counters=counters, gauges=gauges,
                               histograms=hists)

    def sample_gauges(self) -> None:
        """压测过程中周期调用，累计 running / waiting / KV 使用率峰值。"""
        self.snapshot()

    def diff(self, before: MetricsSnapshot,
             after: MetricsSnapshot) -> dict[str, Any] | None:
        """计算区间差值。含服务端侧分位数，与客户端结果并列报告。"""
        if not (before.ok and after.ok):
            return {"available": False,
                    "error": after.error or before.error}

        out: dict[str, Any] = {"available": True, "source": "server_/metrics"}

        for n in _COUNTERS:
            if n in before.counters and n in after.counters:
                out[f"delta_{n.replace('vllm:', '')}"] = round(
                    after.counters[n] - before.counters[n], 3
                )

        q = out.get("delta_prefix_cache_queries_total")
        h = out.get("delta_prefix_cache_hits_total")
        if q and q > 0 and h is not None:
            out["prefix_cache_hit_rate"] = round(h / q, 4)
        elif q == 0:
            out["prefix_cache_hit_rate"] = None
            out["prefix_cache_note"] = "区间内无查询，命中率无定义"

        for k, v in self._peak_gauges.items():
            out[f"peak_{k.replace('vllm:', '')}"] = round(v, 4)
        for k, v in after.gauges.items():
            out[f"final_{k.replace('vllm:', '')}"] = round(v, 4)

        # histogram 分位数：after 减 before 得到本区间分布
        for h_name in _HISTOGRAMS:
            b0 = {le: c for le, c in before.histograms.get(h_name, [])}
            b1 = after.histograms.get(h_name, [])
            if not b1:
                continue
            delta = sorted((le, c - b0.get(le, 0.0)) for le, c in b1)
            short = h_name.replace("vllm:", "").replace("_seconds", "")
            for p in (50, 95, 99):
                r = _hist_pct(delta, p)
                if r is None:
                    continue
                v, upper_only = r
                out[f"server_{short}_p{p}_ms"] = round(v * 1000.0, 2)
                if upper_only:
                    # 标记该值不可当作测量结果，只是分桶上界
                    out[f"server_{short}_p{p}_upper_bound_only"] = True
        return out

    def close(self) -> None:
        self._client.close()
