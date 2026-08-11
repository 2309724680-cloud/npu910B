"""目标服务、SLO 与压测场景的配置模型。

被测对象只要求 OpenAI 兼容接口，任意模型共用同一套配置结构，
切换目标只改 base_url / model / api_key。
"""
from __future__ import annotations

import itertools
import json
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, ClassVar


@dataclass
class TargetConfig:
    """被测服务。"""

    base_url: str
    model: str
    api_key: str = ""
    # 服务标识，仅用于报告标注，不参与请求
    label: str = ""
    # 单请求超时。长输入场景 TTFT 可能到数十秒，read 需留足
    connect_timeout_s: float = 10.0
    read_timeout_s: float = 600.0
    # /metrics 与 /tokenize 是否可用，init 时探测后回填
    has_metrics: bool = False
    has_tokenize: bool = False
    max_model_len: int | None = None

    @property
    def chat_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"

    @property
    def models_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/models"

    @property
    def root_url(self) -> str:
        """去掉 /v1 后缀，/metrics 与 /tokenize 挂在根路径。"""
        b = self.base_url.rstrip("/")
        return b[:-3].rstrip("/") if b.endswith("/v1") else b

    @property
    def metrics_url(self) -> str:
        return f"{self.root_url}/metrics"

    @property
    def tokenize_url(self) -> str:
        return f"{self.root_url}/tokenize"

    def headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h


@dataclass
class SLOConfig:
    """准入线。Goodput 依赖它，未设置时只能给出物理吞吐。

    按 token 长度档位分别设定，不用单一阈值覆盖所有场景。
    理由见 docs/methodology.md 的 SLO and Goodput 一节。
    这里的值是缺省兜底，正式测试须由调用方按档位覆盖。
    """

    ttft_ms: float = 2000.0
    tpot_ms: float = 100.0
    e2e_ms: float | None = None
    # 判定 Stall 的 ITL 阈值（Stall Rate，见 docs/methodology.md）
    stall_itl_ms: float = 500.0

    def per_scenario(self, overrides: dict[str, Any] | None) -> SLOConfig:
        if not overrides:
            return self
        d = asdict(self)
        d.update({k: v for k, v in overrides.items() if k in d})
        return SLOConfig(**d)


@dataclass
class ScenarioSpec:
    """一个压测场景。

    input_tokens / output_tokens 是目标值，实际输入由 tokens.py 逼近；
    实际输出取决于模型何时吐 EOS，用 ignore_eos 可强制定长。

    命名口径：负载维度统一叫 parallelism（并行度），倍率叫 request_multiplier。
    旧配置的 concurrency / requests_per_concurrency 由 from_dict 自动映射，
    见 LEGACY_KEYS。
    """

    name: str
    input_tokens: int
    output_tokens: int
    # closed：固定并行度闭环；open：泊松到达率开环
    mode: str = "closed"
    # 并行度：同时在飞的请求数。parallelism=1 即单请求基线
    parallelism: int = 1
    request_rate: float | None = None  # open 模式下的 req/s
    num_requests: int | None = None
    # 并行任务倍率：一个并行位跑多少个请求，总请求数 = parallelism × 该值。
    # 参考基线表用 10。倍率固定才能保证各并行档的统计样本量随并行度等比放大，
    # 否则低并行档样本过少、分位数不可信。num_requests 显式给出时以它为准。
    request_multiplier: int = 0
    duration_s: float | None = None
    warmup_requests: int = 0
    # prefix cache 控制：unique 全异前缀，shared 共享前缀，mixed 按比例
    prefix_mode: str = "unique"
    shared_prefix_ratio: float = 0.0
    temperature: float = 0.0
    ignore_eos: bool = True
    # 流式开关。非流式下 TTFT / ITL / TPOT 无法定义（拿不到首 token 时刻），
    # 故此处只接受 true；显式写 false 会在 validate 阶段报错而非静默走流式。
    stream: bool = True
    slo: dict[str, Any] | None = None

    # 旧字段名 -> 新字段名。保留是为了不破坏 conf.qwen.json 这类既有配置
    LEGACY_KEYS: ClassVar[dict[str, str]] = {
        "concurrency": "parallelism",
        "requests_per_concurrency": "request_multiplier",
    }
    # 参与矩阵展开的维度：值为 list 时按笛卡尔积展开成多个场景
    MATRIX_KEYS: ClassVar[tuple[str, ...]] = (
        "parallelism", "input_tokens", "output_tokens",
    )

    @property
    def concurrency(self) -> int:
        """旧名兜底。runner / stats 已切到 parallelism，此处供外部脚本读取。"""
        return self.parallelism

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ScenarioSpec:
        """按新旧字段名兼容地构造。

        直接 ScenarioSpec(**d) 遇到拼错的键只会抛裸 TypeError，看不出是哪个
        场景、哪个键错了；压测配置动辄几十行，这里给出可定位的报错。
        """
        d = dict(raw)
        for old, new in cls.LEGACY_KEYS.items():
            if old in d:
                if new in d:
                    raise ValueError(
                        f"场景 {d.get('name', '?')}: {old} 与 {new} 只能给一个"
                    )
                d[new] = d.pop(old)
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(
                f"场景 {d.get('name', '?')}: 无法识别的字段 {sorted(unknown)}；"
                f"可用字段 {sorted(known)}"
            )
        return cls(**d)

    def resolve_requests(self) -> None:
        """把倍率折算成绝对请求数，原地写回 num_requests。

        执行前调用一次，runner 只认 num_requests，不再关心倍率。
        """
        if self.num_requests:
            return
        if self.request_multiplier > 0 and self.mode == "closed":
            self.num_requests = self.parallelism * self.request_multiplier

    def validate(self, max_model_len: int | None) -> list[str]:
        errs = []
        if self.mode not in ("closed", "open"):
            errs.append(f"{self.name}: mode 只能是 closed / open")
        if self.mode == "open" and not self.request_rate:
            errs.append(f"{self.name}: open 模式必须给 request_rate")
        if self.mode == "closed" and self.parallelism < 1:
            errs.append(f"{self.name}: parallelism 需 >= 1")
        if not self.num_requests and not self.duration_s:
            errs.append(f"{self.name}: 需指定 num_requests 或 duration_s")
        if self.prefix_mode not in ("unique", "shared", "mixed"):
            errs.append(f"{self.name}: prefix_mode 非法")
        if not self.stream:
            errs.append(
                f"{self.name}: stream=false 暂不支持——TTFT / ITL / TPOT 依赖流式"
                "逐 chunk 计时，非流式下这三项无定义。如只需 E2E 与吞吐，"
                "请提 issue 单独实现非流式分支，不要静默按流式跑。"
            )
        if max_model_len:
            need = self.input_tokens + self.output_tokens
            if need > max_model_len:
                errs.append(
                    f"{self.name}: input+output={need} 超过 max_model_len={max_model_len}"
                )
        return errs


_DIM_ABBREV = {"parallelism": "p", "input_tokens": "i", "output_tokens": "o"}


def expand_matrix(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """把含 list 的场景定义展开成笛卡尔积。

    parallelism / input_tokens / output_tokens 任一给成 list 就按维度展开，
    命名追加 _p16_i1024_o128 后缀。三个都给单值时原样返回、名字不变，
    所以旧配置的场景名不会被改写。
    """
    d = dict(raw)
    for old, new in ScenarioSpec.LEGACY_KEYS.items():
        if old in d:
            if new in d:
                raise ValueError(
                    f"场景 {d.get('name', '?')}: {old} 与 {new} 只能给一个"
                )
            d[new] = d.pop(old)

    dims: dict[str, list[Any]] = {}
    for k in ScenarioSpec.MATRIX_KEYS:
        v = d.get(k)
        if isinstance(v, (list, tuple)):
            if not v:
                raise ValueError(f"场景 {d.get('name', '?')}: {k} 是空列表")
            dims[k] = list(v)
    if not dims:
        return [d]

    base = d.get("name", "scenario")
    keys = list(dims)
    out: list[dict[str, Any]] = []
    for combo in itertools.product(*(dims[k] for k in keys)):
        s = dict(d)
        s.update(dict(zip(keys, combo, strict=True)))
        suffix = "".join(
            f"_{_DIM_ABBREV[k]}{v}" for k, v in zip(keys, combo, strict=True)
        )
        s["name"] = f"{base}{suffix}"
        out.append(s)
    return out


def build_scenarios(raws: list[dict[str, Any]]) -> list[ScenarioSpec]:
    """原始场景列表 -> 展开并校验后的 ScenarioSpec 列表。

    重名会让报告里两行无法区分，直接报错而不是让它跑完再发现。
    """
    out: list[ScenarioSpec] = []
    for raw in raws:
        for d in expand_matrix(raw):
            out.append(ScenarioSpec.from_dict(d))
    names = [s.name for s in out]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError(f"场景名重复：{dup}")
    return out


@dataclass
class RunConfig:
    target: TargetConfig
    scenarios: list[ScenarioSpec] = field(default_factory=list)
    slo: SLOConfig = field(default_factory=SLOConfig)
    out_dir: str = "results"
    # 采样服务端 /metrics 的间隔，0 表示只在场景前后各取一次
    metrics_interval_s: float = 0.0
    seed: int = 20260809
    # 部署快照与硬件信息。服务端 API 不暴露启动参数与设备状态，
    # 只能由配置声明后原样记入报告；未声明的字段在报告中显示为 [TBD]。
    deployment: dict[str, Any] = field(default_factory=dict)
    hardware: dict[str, Any] = field(default_factory=dict)


def load_config(path: str) -> RunConfig:
    """从 JSON 读配置。api_key 支持 ${ENV_VAR} 展开，避免明文入库。"""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    t = dict(raw["target"])
    key = t.get("api_key", "")
    if key.startswith("${") and key.endswith("}"):
        t["api_key"] = os.environ.get(key[2:-1], "")
    target = TargetConfig(**t)

    slo = SLOConfig(**raw.get("slo", {}))
    scenarios = build_scenarios(raw.get("scenarios", []))
    return RunConfig(
        target=target,
        scenarios=scenarios,
        slo=slo,
        out_dir=raw.get("out_dir", "results"),
        metrics_interval_s=raw.get("metrics_interval_s", 0.0),
        seed=raw.get("seed", 20260809),
        deployment=raw.get("deployment", {}),
        hardware=raw.get("hardware", {}),
    )
