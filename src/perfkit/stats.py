"""分位数、吞吐与 Goodput 聚合。

分位数统一用线性插值（numpy 默认），与 vLLM /metrics 的 histogram 分桶估算
存在方法差异，两者并列报告时需注明来源，不可混用比较。
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from .record import RequestRecord

PCTS = (50, 90, 95, 99)

# 判定吞吐拐点的相对增益阈值：并行度翻档后总吞吐涨幅低于此值即视为已饱和。
# 判据见 docs/methodology.md 的 Saturation 一节。
TPS_KNEE_GAIN = 0.05


def _percentile(sorted_vals: list[float], p: float) -> float:
    """线性插值分位数，与 numpy.percentile 的默认方法一致。

    这里不引入 numpy：全库唯一的数值需求就是这一个函数，
    为它加一个几十 MB 的依赖不值得，且分位数口径必须可审计。
    """
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * p / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_vals[int(pos)]
    return sorted_vals[lo] * (hi - pos) + sorted_vals[hi] * (pos - lo)


def pcts(vals: Sequence[float], ps: Sequence[int] = PCTS) -> dict[str, float]:
    """返回 p50/p90/p95/p99/max/mean/std。空输入返回空字典。"""
    if not vals:
        return {}
    a = sorted(float(v) for v in vals)
    out: dict[str, float] = {f"p{p}": _percentile(a, p) for p in ps}
    mean = math.fsum(a) / len(a)
    out["max"] = a[-1]
    out["min"] = a[0]
    out["mean"] = mean
    # 总体标准差（ddof=0）：样本即全部请求，不做无偏校正
    out["std"] = math.sqrt(math.fsum((x - mean) ** 2 for x in a) / len(a))
    out["count"] = len(a)
    return out


@dataclass
class ScenarioStats:
    """单场景聚合结果。"""

    name: str
    mode: str
    # 字段名保留 concurrency：report.py 与既有 results/*_summary.json 都按此读取，
    # 改名会让历史数据 schema 不兼容。对外统一口径由 parallelism 属性与
    # to_json() 的双名输出提供。
    concurrency: int
    request_rate: float | None

    target_input_tokens: int
    target_output_tokens: int

    total_requests: int
    ok_requests: int
    failed_requests: int
    error_breakdown: dict[str, int]

    # 墙钟窗口：首个请求发出到最后一个响应结束
    wall_duration_s: float

    ttft_ms: dict[str, float]
    tpot_ms: dict[str, float]
    itl_ms: dict[str, float]
    e2e_ms: dict[str, float]
    gen_tps: dict[str, float]

    input_tps: float
    output_tps: float
    rps: float

    goodput_rps: float
    goodput_ratio: float
    slo_used: dict[str, Any]

    stall_rate: float
    mean_input_tokens: float
    mean_output_tokens: float

    # 服务端 /metrics 快照差值，可为空
    server_metrics: dict[str, Any] | None = None

    # 入参回填。报告需与结果同表呈现，否则事后对不上跑的是哪组参数
    request_multiplier: int = 0
    prefix_mode: str = "unique"
    temperature: float = 0.0
    stream: bool = True
    api_url: str = ""
    model: str = ""

    # SLO 分项违约计数。Goodput 为 0 时靠它定位是哪一项指标不达标
    slo_violations: dict[str, int] = field(default_factory=dict)
    # 单请求端到端吞吐 output_tokens / e2e 的分位数，对齐基线表「平均吞吐」
    per_request_tps: dict[str, float] = field(default_factory=dict)
    # 复现要素：预热数、是否定长输出、该档配置的时长上限（None 为按请求数跑满）
    warmup_requests: int = 0
    ignore_eos: bool = True
    duration_cap_s: float | None = None

    @property
    def parallelism(self) -> int:
        """对外统一口径。内部字段仍名为 concurrency，见该字段注释。"""
        return self.concurrency

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        # 双名输出：新消费方读 parallelism，历史脚本与旧 summary 仍能读 concurrency
        d["parallelism"] = self.concurrency
        d["requests_per_concurrency"] = self.request_multiplier
        return d


def aggregate(
    name: str,
    mode: str,
    parallelism: int,
    request_rate: float | None,
    target_in: int,
    target_out: int,
    records: list[RequestRecord],
    slo_ttft_ms: float,
    slo_tpot_ms: float,
    slo_e2e_ms: float | None,
    stall_itl_ms: float,
    server_metrics: dict[str, Any] | None = None,
    request_multiplier: int = 0,
    prefix_mode: str = "unique",
    temperature: float = 0.0,
    stream: bool = True,
    api_url: str = "",
    model: str = "",
    warmup_requests: int = 0,
    ignore_eos: bool = True,
    duration_cap_s: float | None = None,
) -> ScenarioStats:
    ok = [r for r in records if r.status == "ok"]
    bad = [r for r in records if r.status != "ok"]

    errs: dict[str, int] = {}
    for r in bad:
        k = r.error_type or "unknown"
        errs[k] = errs.get(k, 0) + 1

    # 墙钟窗口只按成功请求算会高估吞吐，这里用全部请求的时间跨度
    starts = [r.send_time for r in records if r.send_time]
    ends = [r.end_time for r in records if r.end_time is not None]
    wall = (max(ends) - min(starts)) if starts and ends else 0.0

    itl_all: list[float] = []
    for r in ok:
        itl_all.extend(r.itl_ms_list)

    in_tok = sum(r.input_tokens for r in ok)
    out_tok = sum(r.output_tokens for r in ok)

    good = [r for r in ok if r.meets_slo(slo_ttft_ms, slo_tpot_ms, slo_e2e_ms)]
    stalled = [r for r in ok if r.stalled(stall_itl_ms)]

    # 违约归因：一条请求可能同时违反多项，各项独立计数，故总和可能大于请求数
    viol: dict[str, int] = {}
    for r in records:
        for k in r.slo_violations(slo_ttft_ms, slo_tpot_ms, slo_e2e_ms):
            viol[k] = viol.get(k, 0) + 1

    req_tps = [
        r.output_tokens / (r.e2e_ms / 1000.0)
        for r in ok if r.e2e_ms and r.e2e_ms > 0
    ]

    def safe_div(a: float, b: float) -> float:
        return a / b if b > 0 else 0.0

    return ScenarioStats(
        warmup_requests=warmup_requests,
        ignore_eos=ignore_eos,
        duration_cap_s=duration_cap_s,
        name=name,
        mode=mode,
        concurrency=parallelism,
        request_rate=request_rate,
        target_input_tokens=target_in,
        target_output_tokens=target_out,
        total_requests=len(records),
        ok_requests=len(ok),
        failed_requests=len(bad),
        error_breakdown=errs,
        wall_duration_s=wall,
        ttft_ms=pcts([r.ttft_ms for r in ok if r.ttft_ms is not None]),
        tpot_ms=pcts([r.tpot_ms for r in ok if r.tpot_ms is not None]),
        itl_ms=pcts(itl_all),
        e2e_ms=pcts([r.e2e_ms for r in ok if r.e2e_ms is not None]),
        gen_tps=pcts([r.gen_tps for r in ok if r.gen_tps is not None]),
        input_tps=safe_div(in_tok, wall),
        output_tps=safe_div(out_tok, wall),
        rps=safe_div(len(ok), wall),
        goodput_rps=safe_div(len(good), wall),
        goodput_ratio=safe_div(len(good), len(records)),
        slo_used={
            "ttft_ms": slo_ttft_ms,
            "tpot_ms": slo_tpot_ms,
            "e2e_ms": slo_e2e_ms,
            "stall_itl_ms": stall_itl_ms,
        },
        stall_rate=safe_div(len(stalled), len(ok)),
        mean_input_tokens=safe_div(in_tok, len(ok)),
        mean_output_tokens=safe_div(out_tok, len(ok)),
        server_metrics=server_metrics,
        request_multiplier=request_multiplier,
        prefix_mode=prefix_mode,
        temperature=temperature,
        stream=stream,
        api_url=api_url,
        model=model,
        slo_violations=viol,
        per_request_tps=pcts(req_tps),
    )


def find_saturation(steps: list[ScenarioStats]) -> dict[str, Any]:
    """从并行度阶梯里找饱和点。判据见 docs/methodology.md。

    三条判据分别独立报告，取最早出现的作为饱和点：
      1. Output TPS 不再上升（相对前一阶梯增幅 < 5%）
      2. Goodput 开始下降
      3. P95 TTFT 或 P95 TPOT 越过 SLO

    仅一个阶梯时无法判定，返回 reason=insufficient_steps。

    只在**相同 (input, output) 长度**的场景之间比较：判据 1 和 2 的前提是
    「其他条件不变、仅并行度递增」。长度不同的场景 TPS 与 Goodput 本就不可比
    （长输入的 prefill 更重、长输出摊薄 TTFT 占比），混在一起排序会把
    「换了长度档导致的吞吐下降」误判成「并行度到顶」。
    矩阵扫描下这种混合是常态，故按长度分组后取阶梯最长的那组。
    """
    if len(steps) < 2:
        return {"saturated": False, "reason": "insufficient_steps"}

    groups: dict[tuple[int, int], list[ScenarioStats]] = {}
    for st in steps:
        groups.setdefault(
            (st.target_input_tokens, st.target_output_tokens), []
        ).append(st)
    # 阶梯最长的组作为容量判定依据；并列时取输入更短的（更接近典型对话负载）
    key = max(groups, key=lambda k: (len({x.concurrency for x in groups[k]}), -k[0]))
    s = sorted(groups[key], key=lambda x: x.concurrency)
    skipped = [
        {"input_tokens": k[0], "output_tokens": k[1],
         "parallelism": sorted({x.concurrency for x in groups[k]})}
        for k in groups if k != key
    ]
    if len(s) < 2:
        return {
            "saturated": False,
            "reason": "insufficient_steps_per_length_group",
            "basis_input_tokens": key[0],
            "basis_output_tokens": key[1],
            "excluded_groups": skipped,
        }
    tps_knee = goodput_knee = slo_knee = None

    for i in range(1, len(s)):
        prev, cur = s[i - 1], s[i]
        if (
            tps_knee is None
            and prev.output_tps > 0
            and (cur.output_tps - prev.output_tps) / prev.output_tps < TPS_KNEE_GAIN
        ):
            tps_knee = cur.concurrency
        if goodput_knee is None and cur.goodput_rps < prev.goodput_rps:
            goodput_knee = cur.concurrency

    for cur in s:
        if slo_knee is not None:
            break
        t95 = cur.ttft_ms.get("p95")
        p95 = cur.tpot_ms.get("p95")
        lim = cur.slo_used
        if (t95 is not None and t95 > lim["ttft_ms"]) or (
            p95 is not None and p95 > lim["tpot_ms"]
        ):
            slo_knee = cur.concurrency

    cands = [c for c in (tps_knee, goodput_knee, slo_knee) if c is not None]
    best = max(s, key=lambda x: x.goodput_rps)

    return {
        "saturated": bool(cands),
        "saturation_concurrency": min(cands) if cands else None,
        "tps_plateau_at": tps_knee,
        "goodput_decline_at": goodput_knee,
        "slo_violation_at": slo_knee,
        "max_goodput_rps": best.goodput_rps,
        "max_goodput_at_concurrency": best.concurrency,
        # 判定依据的长度档。不写出来的话，矩阵扫描的报告里看不出容量结论
        # 是在哪个 input/output 组合下得到的，换档后无法复现
        "basis_input_tokens": key[0],
        "basis_output_tokens": key[1],
        "basis_parallelism_steps": [x.concurrency for x in s],
        "excluded_groups": skipped,
        # §4：限流取最大稳定 Goodput 的 70%~80%，不取饱和点
        "recommended_rate_limit_rps": round(best.goodput_rps * 0.8, 3),
        "recommended_rate_limit_note": "最大稳定 Goodput 的 80%，需与调用方确认后生效",
    }
