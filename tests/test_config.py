"""配置解析、矩阵展开与校验的回归测试。

这三处是压测跑起来之前唯一的拦网：配置错了要在启动瞬间失败，
而不是跑几十分钟后才发现场景重名、长度超限或静默走了非流式。
"""
from __future__ import annotations

import pytest

from perfkit.config import (
    ScenarioSpec,
    SLOConfig,
    TargetConfig,
    build_scenarios,
    expand_matrix,
)


def test_target_urls_strip_v1_for_root_endpoints():
    t = TargetConfig(base_url="http://example.invalid:8000/v1", model="m")
    assert t.chat_url == "http://example.invalid:8000/v1/chat/completions"
    assert t.models_url == "http://example.invalid:8000/v1/models"
    # /metrics 与 /tokenize 挂在根路径，不在 /v1 下
    assert t.metrics_url == "http://example.invalid:8000/metrics"
    assert t.tokenize_url == "http://example.invalid:8000/tokenize"


def test_target_urls_without_v1_suffix():
    t = TargetConfig(base_url="http://example.invalid:8000/", model="m")
    assert t.chat_url == "http://example.invalid:8000/chat/completions"
    assert t.metrics_url == "http://example.invalid:8000/metrics"


def test_headers_omit_auth_when_no_key():
    assert "Authorization" not in TargetConfig(base_url="u", model="m").headers()
    h = TargetConfig(base_url="u", model="m", api_key="k").headers()
    assert h["Authorization"] == "Bearer k"


def test_legacy_keys_are_mapped():
    s = ScenarioSpec.from_dict({
        "name": "x", "input_tokens": 128, "output_tokens": 32,
        "concurrency": 8, "requests_per_concurrency": 10,
    })
    assert s.parallelism == 8
    assert s.request_multiplier == 10
    assert s.concurrency == 8


def test_legacy_and_new_key_together_is_an_error():
    with pytest.raises(ValueError, match="只能给一个"):
        ScenarioSpec.from_dict({
            "name": "x", "input_tokens": 1, "output_tokens": 1,
            "concurrency": 8, "parallelism": 8,
        })


def test_unknown_field_names_the_scenario():
    with pytest.raises(ValueError, match="parallelizm"):
        ScenarioSpec.from_dict({
            "name": "bad", "input_tokens": 1, "output_tokens": 1,
            "parallelizm": 8,
        })


def test_resolve_requests_from_multiplier():
    s = ScenarioSpec(name="x", input_tokens=1, output_tokens=1,
                     parallelism=8, request_multiplier=10)
    s.resolve_requests()
    assert s.num_requests == 80


def test_explicit_num_requests_wins_over_multiplier():
    s = ScenarioSpec(name="x", input_tokens=1, output_tokens=1,
                     parallelism=8, request_multiplier=10, num_requests=5)
    s.resolve_requests()
    assert s.num_requests == 5


def test_open_mode_ignores_multiplier():
    s = ScenarioSpec(name="x", input_tokens=1, output_tokens=1,
                     mode="open", request_rate=1.0, request_multiplier=10)
    s.resolve_requests()
    assert s.num_requests is None


def test_single_valued_scenario_keeps_its_name():
    # 旧配置不该因为引入矩阵展开而被改名
    raw = {"name": "keep", "input_tokens": 1024, "output_tokens": 128,
           "parallelism": 4}
    assert expand_matrix(raw) == [raw]


def test_matrix_expands_cartesian_with_named_suffix():
    out = expand_matrix({
        "name": "sweep", "parallelism": [1, 4],
        "input_tokens": [1024], "output_tokens": [128, 512],
    })
    assert len(out) == 4
    assert [d["name"] for d in out] == [
        "sweep_p1_i1024_o128", "sweep_p1_i1024_o512",
        "sweep_p4_i1024_o128", "sweep_p4_i1024_o512",
    ]


def test_empty_matrix_list_is_an_error():
    with pytest.raises(ValueError, match="空列表"):
        expand_matrix({"name": "x", "parallelism": [],
                       "input_tokens": 1, "output_tokens": 1})


def test_duplicate_scenario_names_are_rejected():
    # 重名会让报告里两行无法区分
    with pytest.raises(ValueError, match="重复"):
        build_scenarios([
            {"name": "dup", "input_tokens": 1, "output_tokens": 1},
            {"name": "dup", "input_tokens": 2, "output_tokens": 2},
        ])


def test_validate_rejects_non_streaming():
    s = ScenarioSpec(name="x", input_tokens=1, output_tokens=1,
                     num_requests=1, stream=False)
    assert any("stream=false" in e for e in s.validate(None))


def test_validate_requires_rate_in_open_mode():
    s = ScenarioSpec(name="x", input_tokens=1, output_tokens=1,
                     mode="open", num_requests=1)
    assert any("request_rate" in e for e in s.validate(None))


def test_validate_rejects_exceeding_max_model_len():
    s = ScenarioSpec(name="x", input_tokens=1000, output_tokens=100,
                     num_requests=1)
    assert any("max_model_len" in e for e in s.validate(1024))
    assert s.validate(4096) == []


def test_validate_requires_a_stop_condition():
    s = ScenarioSpec(name="x", input_tokens=1, output_tokens=1)
    assert any("num_requests" in e for e in s.validate(None))


def test_slo_per_scenario_override_is_partial_and_non_mutating():
    base = SLOConfig(ttft_ms=2000.0, tpot_ms=100.0)
    got = base.per_scenario({"ttft_ms": 10000.0, "unknown": 1})
    assert got.ttft_ms == 10000.0
    assert got.tpot_ms == 100.0        # 未覆盖的沿用基线
    assert base.ttft_ms == 2000.0      # 原对象不被改写
    assert base.per_scenario(None) is base
