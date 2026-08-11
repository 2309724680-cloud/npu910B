"""单请求原始记录与延迟指标计算。

字段定义见 docs/methodology.md。耗时全部用 time.perf_counter() 单调时钟计算，
send_time 另存一个 wall clock 供跨端对齐，两者不混用。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RequestRecord:
    request_id: str
    model: str
    workload_class: str

    # wall clock（ISO8601），仅用于与服务端日志对齐
    send_wall: str = ""
    # 单调时钟原始读数（秒），耗时计算只用这三个
    send_time: float = 0.0
    first_token_time: float | None = None
    end_time: float | None = None

    input_tokens: int = 0
    output_tokens: int = 0
    # 服务端 usage 回报值，与本地估算并存，不一致时以此为准
    usage_prompt_tokens: int | None = None
    usage_completion_tokens: int | None = None

    status: str = "pending"          # ok / error / timeout
    error_type: str | None = None
    error_detail: str | None = None
    http_status: int | None = None
    finish_reason: str | None = None

    cache_hit: bool | None = None
    instance: str | None = None
    node: str | None = None
    mtp_acceptance: float | None = None

    # 每个 token 到达的单调时刻，用于 ITL 与 Stall 判定
    token_times: list[float] = field(default_factory=list)
    # 场景标注，报告分组用
    target_input_tokens: int = 0
    target_output_tokens: int = 0
    concurrency: int = 0

    @property
    def ttft_ms(self) -> float | None:
        if self.first_token_time is None:
            return None
        return (self.first_token_time - self.send_time) * 1000.0

    @property
    def e2e_ms(self) -> float | None:
        if self.end_time is None:
            return None
        return (self.end_time - self.send_time) * 1000.0

    @property
    def tpot_ms(self) -> float | None:
        """首 token 之后的平均每 token 耗时（TPOT，见 docs/methodology.md）。

        分母是 output_tokens - 1：首 token 的耗时归 TTFT，不重复计入。
        输出仅 1 个 token 时 TPOT 无定义，返回 None 而不是 0。

        分子用最后一个含内容 chunk 的到达时刻，而非 end_time——end_time 还包含
        末尾 usage chunk 与连接关闭的开销，那部分不属于解码时间，
        短输出场景下会把 TPOT 抬高十几个百分点。
        """
        if self.first_token_time is None:
            return None
        n = self.output_tokens
        if n < 2:
            return None
        last = self.token_times[-1] if self.token_times else self.end_time
        if last is None:
            return None
        return (last - self.first_token_time) * 1000.0 / (n - 1)

    @property
    def itl_ms_list(self) -> list[float]:
        """相邻 chunk 间隔。首 chunk 不产生 ITL。

        口径说明：这里测的是 SSE chunk 到达间隔，不严格等于 token 间隔。
        一个 chunk 可能携带多个 token（实测 32 token 的响应只有 31 个含内容
        的 chunk），因此 ITL 样本数通常略少于 output_tokens - 1。
        用于观察卡顿和抖动是准确的；要精确的 per-token 间隔需看服务端
        inter_token_latency histogram。
        """
        ts = self.token_times
        if len(ts) < 2:
            return []
        return [(ts[i] - ts[i - 1]) * 1000.0 for i in range(1, len(ts))]

    @property
    def chunk_count(self) -> int:
        """含内容的 chunk 数。与 output_tokens 的差值反映 chunk 聚合程度。"""
        return len(self.token_times)

    @property
    def gen_tps(self) -> float | None:
        """单请求解码吞吐（Generation TPS，见 docs/methodology.md）。"""
        if self.first_token_time is None or self.end_time is None:
            return None
        dur = self.end_time - self.first_token_time
        if dur <= 0 or self.output_tokens < 2:
            return None
        return (self.output_tokens - 1) / dur

    def stalled(self, threshold_ms: float) -> bool:
        itls = self.itl_ms_list
        return bool(itls) and max(itls) > threshold_ms

    def meets_slo(self, ttft_ms: float, tpot_ms: float,
                  e2e_ms: float | None = None) -> bool:
        """Goodput 判定：成功 + TTFT 达标 + TPOT 达标。

        TPOT 无定义（输出 < 2 token）时不做 TPOT 约束，
        否则定长 128 输出的短对话会被误判为不达标。
        """
        if self.status != "ok":
            return False
        t = self.ttft_ms
        if t is None or t > ttft_ms:
            return False
        p = self.tpot_ms
        if p is not None and p > tpot_ms:
            return False
        if e2e_ms is not None:
            e = self.e2e_ms
            if e is None or e > e2e_ms:
                return False
        return True

    def slo_violations(self, ttft_ms: float, tpot_ms: float,
                       e2e_ms: float | None = None) -> list[str]:
        """列出这条请求违反了哪几项 SLO，供报告归因。

        `meets_slo` 只给出布尔值，无法回答「Goodput 为 0 但成功率 100%」这类
        问题。分项计数才能指出是 TTFT 还是 TPOT 拖垮了 Goodput。
        """
        if self.status != "ok":
            return ["failed"]
        v: list[str] = []
        t = self.ttft_ms
        if t is None or t > ttft_ms:
            v.append("ttft")
        p = self.tpot_ms
        if p is not None and p > tpot_ms:
            v.append("tpot")
        if e2e_ms is not None:
            e = self.e2e_ms
            if e is None or e > e2e_ms:
                v.append("e2e")
        return v

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        # token_times 体积大且只用于聚合，落盘时换成 ITL 摘要
        itls = self.itl_ms_list
        d.pop("token_times", None)
        d["ttft_ms"] = self.ttft_ms
        d["tpot_ms"] = self.tpot_ms
        d["e2e_ms"] = self.e2e_ms
        d["gen_tps"] = self.gen_tps
        d["chunk_count"] = self.chunk_count
        d["itl_count"] = len(itls)
        d["itl_max_ms"] = max(itls) if itls else None
        d["itl_mean_ms"] = sum(itls) / len(itls) if itls else None
        return d


class JsonlWriter:
    """逐条落盘，压测中断也不丢已完成的数据。"""

    def __init__(self, path: str):
        self.path = path
        # 句柄的生命周期就是本对象的生命周期，由 __exit__ / close 负责释放；
        # 压测全程持续追加写，不能每条记录开关一次文件
        self._f = open(path, "a", encoding="utf-8")  # noqa: SIM115

    def write(self, rec: RequestRecord) -> None:
        self._f.write(json.dumps(rec.to_json(), ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
