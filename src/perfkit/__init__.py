"""perfkit — a benchmark toolkit for OpenAI-compatible LLM inference servers.

Works against anything exposing `/v1/chat/completions`: vLLM, SGLang, TGI,
llama.cpp server, or a hosted API. Switching targets is a config change only.
Validated on both NVIDIA GPUs and Ascend NPUs.

Measures TTFT, TPOT, ITL, throughput and Goodput under closed-loop or
open-loop (Poisson) load, sweeps concurrency and sequence lengths, scrapes
server-side vLLM `/metrics`, and emits Markdown + CSV + JSON reports.

See docs/methodology.md for metric definitions and measurement caveats.
"""
from .config import RunConfig, ScenarioSpec, SLOConfig, TargetConfig, load_config
from .metrics import MetricsCollector
from .record import JsonlWriter, RequestRecord
from .report import save_all
from .runner import ScenarioRunner, probe_target
from .stats import ScenarioStats, aggregate, find_saturation
from .tokens import PromptFactory, TokenCounter

__version__ = "0.1.0"

__all__ = [
    "RunConfig", "SLOConfig", "ScenarioSpec", "TargetConfig", "load_config",
    "MetricsCollector", "JsonlWriter", "RequestRecord", "save_all",
    "ScenarioRunner", "probe_target", "ScenarioStats", "aggregate",
    "find_saturation", "PromptFactory", "TokenCounter",
]
