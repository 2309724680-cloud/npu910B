"""输出报告：JSON 全量 + Markdown 摘要 + CSV 曲线数据。

Markdown 供人读，CSV 供画四条标准曲线，JSON 是唯一权威数据源。
"""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from typing import Any

from .stats import ScenarioStats, find_saturation


def _fmt(v: Any, digits: int = 1) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _cell(d: dict[str, float], key: str, digits: int = 1) -> str:
    return _fmt(d.get(key), digits) if d else "-"


def _sec(d: dict[str, float], key: str, digits: int = 3) -> str:
    """毫秒字典按秒输出，对齐基线表的秒制列。"""
    if not d:
        return "-"
    v = d.get(key)
    return "-" if v is None else f"{v / 1000.0:.{digits}f}"


def _srv(sm: dict[str, Any], key: str, digits: int = 1) -> str:
    """服务端分位数取值。落在首个分桶内时显示为 `<= X`，避免当成测量值。"""
    v = sm.get(key)
    if v is None:
        return "-"
    flag = key.replace("_ms", "_upper_bound_only")
    prefix = "<= " if sm.get(flag) else ""
    return f"{prefix}{_fmt(v, digits)}"


def _tbd(d: dict[str, Any], key: str) -> str:
    """未声明的部署 / 硬件字段标 [TBD]，不留空也不猜测。"""
    v = (d or {}).get(key)
    if v is None or v == "":
        return "[TBD]"
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def write_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_curves_csv(path: str, steps: list[ScenarioStats]) -> None:
    """并发阶梯曲线数据。四列对应四条标准曲线。"""
    cols = [
        "scenario", "mode", "concurrency", "request_rate",
        "input_tokens", "output_tokens",
        "ok", "failed",
        "output_tps", "input_tps", "rps", "goodput_rps", "goodput_ratio",
        "ttft_p50", "ttft_p95", "ttft_p99",
        "tpot_p50", "tpot_p95", "tpot_p99",
        "itl_p95", "itl_max", "e2e_p95",
        "stall_rate",
        "server_waiting_peak", "server_kv_peak", "prefix_hit_rate",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for s in steps:
            sm = s.server_metrics or {}
            w.writerow([
                s.name, s.mode, s.concurrency, s.request_rate or "",
                s.target_input_tokens, s.target_output_tokens,
                s.ok_requests, s.failed_requests,
                round(s.output_tps, 2), round(s.input_tps, 2),
                round(s.rps, 3), round(s.goodput_rps, 3), round(s.goodput_ratio, 4),
                _cell(s.ttft_ms, "p50"), _cell(s.ttft_ms, "p95"), _cell(s.ttft_ms, "p99"),
                _cell(s.tpot_ms, "p50"), _cell(s.tpot_ms, "p95"), _cell(s.tpot_ms, "p99"),
                _cell(s.itl_ms, "p95"), _cell(s.itl_ms, "max"), _cell(s.e2e_ms, "p95"),
                round(s.stall_rate, 4),
                sm.get("peak_num_requests_waiting", ""),
                sm.get("peak_kv_cache_usage_perc", ""),
                sm.get("prefix_cache_hit_rate", ""),
            ])


def render_markdown(
    env: dict[str, Any],
    steps: list[ScenarioStats],
    saturation: dict[str, Any],
) -> str:
    L: list[str] = []
    A = L.append

    # 标题用真实模型名：env["model"] 是 --served-model-name 别名（如 dsv4），
    # 作为交付物标题看不出测的是什么模型。真名由 deployment.model_name 声明。
    _dep0 = env.get("deployment") or {}
    _title_model = _dep0.get("model_name") or env.get("model", "-")
    A(f"# 性能测试报告：{_title_model}")
    A("")
    A(f"生成时间：{datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}")
    A("")

    A("## 1. 报告定位")
    A("")
    A("**本报告是推理服务容量评估报告，不是单纯的 TPS 跑分。**"
      "目标是回答「这套服务能同时承载多少用户、且每个用户的体验仍可接受」，"
      "而不是「这张卡峰值能吐多少 token」。"
      "两者在过载区会给出完全相反的结论：吞吐仍在上升时，"
      "用户体验可能已经崩掉（见 §5.3、§6）。")
    A("")
    A("### 1.1 指标优先级")
    A("")
    A("结论冲突时按此顺序取舍，**顺序不可调换**：")
    A("")
    A("| 优先级 | 指标 | 为什么在这个位置 |")
    A("|---|---|---|")
    A("| 1 | **Goodput** | 唯一同时包含「正确返回」和「足够快」的指标，"
      "直接对应可交付的用户请求数 |")
    A("| 2 | **TTFT P95** | 首字等待是用户最敏感的体感项；"
      "取 P95 而非均值，因为过载时均值会被少数快请求拉平 |")
    A("| 3 | **TPOT** | 决定出字流畅度。TTFT 达标但 TPOT 超标表现为「开头快、后面卡」 |")
    A("| 4 | **TPS** | 集群成本核算指标。**不可用于选择工作点** |")
    A("")
    A("### 1.2 测试用途")
    A("")
    A("- **SDK 验证**：确认采集链路、指标口径、报告产出可用（当前阶段）")
    A("- **模型服务基准测试**：为同一模型的不同部署参数、不同硬件建立可比基线")
    A("- **容量规划**：给出单节点可承载的并发数与请求速率")
    A("- **限流策略制定**：把 Goodput 上限折算成网关限流阈值（见 §9）")
    A("")

    A("## 2. 测试环境快照")
    A("")
    A("| 项 | 值 |")
    A("|---|---|")
    # 模型行单独渲染：env["model"] 只是 --served-model-name 别名，
    # 直接打出来会让读者以为被测模型就叫那个名字（详见 §3.1）
    A(f"| 被测模型 | **{_title_model}** |")
    if _title_model != env.get("model"):
        A(f"| API 服务名（请求体 `model` 字段） | `{env.get('model', '-')}` |")
    for k, label in (
        ("label", "部署标注"),
        ("base_url", "base_url"),
        ("max_model_len", "max_model_len"),
        ("has_metrics", "has_metrics"),
        ("has_tokenize", "has_tokenize"),
        ("token_source", "token_source"),
        ("sdk_version", "sdk_version"),
    ):
        if k in env:
            A(f"| {label} | {env[k]} |")
    A("")

    if env.get("warnings"):
        A("> 数据可信度提示：")
        for w in env["warnings"]:
            A(f"> - {w}")
        A("")

    dep = env.get("deployment") or {}
    A("## 3. 模型与服务部署快照")
    A("")
    A("推理服务的启动参数直接决定性能上限，**换参数即换基线**。"
      "下列字段服务端 API 不暴露，由配置声明后原样记入；"
      "`[TBD]` 表示未声明，跨轮对比前必须补齐。")
    A("")
    A("### 3.1 模型信息")
    A("")
    A("| 项 | 值 | 说明 |")
    A("|---|---|---|")
    A(f"| **模型** | {_tbd(dep, 'model_name')} | 实际被测模型（权重目录标识） |")
    A(f"| 上游模型 | {_tbd(dep, 'base_model')} | 量化前的原始模型 |")
    A(f"| API 服务名 | `{env.get('model', '[TBD]')}` | "
      "`--served-model-name` 别名，请求体 `model` 字段用它，**不是模型标识** |")
    A(f"| 权重路径 | {_tbd(dep, 'weights_path')} | |")
    A(f"| 参数量 | {_tbd(dep, 'param_count')} | |")
    A(f"| 架构 | {_tbd(dep, 'architecture')} | |")
    A(f"| 精度 | {_tbd(dep, 'dtype')} | |")
    A(f"| 量化方式 | {_tbd(dep, 'quantization')} | 如 w8a8 / awq / 无 |")
    A(f"| 最大上下文 | {env.get('max_model_len') or '[TBD]'} | "
      "服务端 `--max-model-len` 生效值 |")
    A(f"| 模型原生上下文 | {_tbd(dep, 'native_max_position')} | "
      "权重 config 声明值；服务端可低于它启动 |")
    A(f"| tokenizer | {_tbd(dep, 'tokenizer')} | 影响 token 计数口径 |")
    A("")
    A("### 3.2 服务启动参数")
    A("")
    A("| 参数 | 值 | 对性能的影响 |")
    A("|---|---|---|")
    A(f"| `tensor_parallel_size` | {_tbd(dep, 'tensor_parallel_size')} | "
      "切分卡数。影响单卡显存占用与卡间通信量 |")
    mml = dep.get("max_model_len") or env.get("max_model_len") or "[TBD]"
    A(f"| `max_model_len` | {mml} | "
      "上限越大，单请求预留 KV cache 越多，可并发数越少 |")
    A(f"| `gpu_memory_utilization` | {_tbd(dep, 'gpu_memory_utilization')} | "
      "KV cache 池大小的直接决定项，进而决定并发容量 |")
    A(f"| `max_num_seqs` | {_tbd(dep, 'max_num_seqs')} | "
      "单批最大请求数。**并发超过此值即开始排队**，是 TTFT 拐点的主因 |")
    A(f"| `max_num_batched_tokens` | {_tbd(dep, 'max_num_batched_tokens')} | "
      "单批 token 上限，长输入场景下先于 `max_num_seqs` 触顶 |")
    A(f"| `quantization` | {_tbd(dep, 'quantization')} | 权重体积与算子路径 |")
    A(f"| `enforce_eager` | {_tbd(dep, 'enforce_eager')} | "
      "`true` 关闭图优化，延迟显著变高，**生产测试须为 false** |")
    A(f"| `enable_prefix_caching` | {_tbd(dep, 'enable_prefix_caching')} | "
      "开启后共享前缀请求的 TTFT 大幅下降，需与 §4 的 prefix 策略对齐解读 |")
    A(f"| 额外参数 | {_tbd(dep, 'extra_args')} | 如 MTP / EP / 调度策略 |")
    A("")
    if dep.get("launch_command"):
        A("完整启动命令：")
        A("")
        A("```bash")
        A(str(dep["launch_command"]))
        A("```")
        A("")

    A("## 4. 压测入参")
    A("")
    A("每档的负载参数。「倍率」为一个并行位承担的请求数，"
      "总请求数 = 并行度 × 倍率；倍率列为 `-` 表示该档用了固定请求数覆盖。")
    A("")
    A("| 场景 | 并行度 | 倍率 | 总请求数 | 输入长度(目标/实测均值) | "
      "输出长度(目标/实测均值) | 温度 | 流式 | Prefix 模式 |")
    A("|---|---|---|---|---|---|---|---|---|")
    for s in steps:
        load = s.parallelism if s.mode == "closed" else f"{s.request_rate}/s(开环)"
        mult = s.request_multiplier or "-"
        A(
            f"| {s.name} | {load} | {mult} | {s.total_requests} | "
            f"{s.target_input_tokens} / {s.mean_input_tokens:.0f} | "
            f"{s.target_output_tokens} / {s.mean_output_tokens:.0f} | "
            f"{s.temperature} | {'是' if s.stream else '否'} | "
            f"{s.prefix_mode} |"
        )
    A("")

    A("### 4.1 为什么各档请求数不同")
    A("")
    A("**总请求数 = 并行度 × 倍率**，所以高并发档的请求数天然更多。"
      "这不是样本量不一致，而是刻意保证**每个并行位承担相同的请求数**："
      "若各档都用固定总数，低并发档每个并行位要跑很多轮、高并发档只跑一轮，"
      "两者的稳态时长和调度器行为都不可比。"
      "按倍率折算后，各档的稳态持续时间大致相当，横向对比才成立。")
    A("")

    A("### 4.2 分场景入参明细")
    A("")
    A("逐场景列出目标值与实测值。**target 与 actual 的差异是正常的**："
      "输入长度由 prompt 生成器按 token 数逼近，受 tokenizer 分词边界限制"
      "无法精确命中；输出长度在 `ignore_eos=true` 下应严格等于目标值，"
      "若不等说明服务端未支持该字段，此时各请求输出长度不一，吞吐横向对比失效。")
    A("")
    for s in steps:
        load = (f"{s.parallelism}" if s.mode == "closed"
                else f"{s.request_rate}/s(开环)")
        mult = s.request_multiplier or "-"
        in_gap = s.mean_input_tokens - s.target_input_tokens
        out_gap = s.mean_output_tokens - s.target_output_tokens
        A(f"**{s.name}**")
        A("")
        A("| 参数 | 值 |")
        A("|---|---|")
        A(f"| parallelism | {load} |")
        A(f"| request multiplier | {mult} |")
        A(f"| total requests | {s.total_requests} |")
        A(f"| input tokens target | {s.target_input_tokens} |")
        A(f"| input tokens actual | {s.mean_input_tokens:.0f}"
          f"（{in_gap:+.0f}） |")
        A(f"| output tokens target | {s.target_output_tokens} |")
        A(f"| output tokens actual | {s.mean_output_tokens:.0f}"
          f"（{out_gap:+.0f}） |")
        A(f"| mode | {s.mode} |")
        A(f"| stream | {'true' if s.stream else 'false'} |")
        A(f"| prefix mode | {s.prefix_mode} |")
        A(f"| temperature | {s.temperature} |")
        A(f"| ignore_eos | {'true' if s.ignore_eos else 'false'} |")
        A("")

    A("### 4.3 复现要素")
    A("")
    A("| 项 | 值 | 说明 |")
    A("|---|---|---|")
    steps_dur = ", ".join(
        f"{s.name}={s.wall_duration_s:.1f}s" for s in steps
    ) or "-"
    A(f"| 实际持续时长 | {steps_dur} | 各档墙钟时长，含收尾请求 |")
    caps = {s.duration_cap_s for s in steps}
    cap_txt = (", ".join(f"{c:.0f}s" for c in sorted(c for c in caps if c))
               if any(caps) else "未设置（按请求数跑满）")
    A(f"| 时长上限 | {cap_txt} | 设置后先到者停止 |")
    warm = ", ".join(f"{s.name}={s.warmup_requests}" for s in steps) or "-"
    A(f"| warmup 请求数 | {warm} | 不计入统计，用于触发图编译与缓存预热 |")
    pmodes = sorted({s.prefix_mode for s in steps})
    A(f"| prompt 生成策略 | {', '.join(pmodes) or '-'} | "
      "`unique` 每请求独立随机内容；`shared` 共享固定前缀 |")
    A(f"| prefix cache 策略 | {'; '.join(pmodes) or '-'}"
      f"（服务端开关见 §3.2） | "
      "`unique` 规避命中以测冷启动上限；`shared` 用于量化缓存收益 |")
    eos = {s.ignore_eos for s in steps}
    eos_txt = ("是（ignore_eos=true，输出定长）" if eos == {True}
               else "否（由模型决定）" if eos == {False} else "各档不同，见 CSV")
    A(f"| 固定输出长度 | {eos_txt} | 定长可消除输出长度差异对吞吐的干扰 |")
    A(f"| random seed | {env.get('seed', '[TBD]')} | "
      "固定后 prompt 内容与长度分布可复现 |")
    A(f"| token 计数来源 | {env.get('token_source', '[TBD]')} | "
      "`server_/tokenize` 与服务端计费口径一致；`char_estimate` 为估算 |")
    A("")

    A("## 5. 延迟与吞吐结果")
    A("")

    A("### 5.1 指标含义")
    A("")
    A("| 指标 | 全称 | 含义 | 决定什么 |")
    A("|---|---|---|---|")
    A("| **TTFT** | Time To First Token | 从发出请求到收到第一个输出 token 的耗时，"
      "包含排队 + prefill。| 用户按下回车后的等待感。对话类产品最敏感的指标。 |")
    A("| **TPOT** | Time Per Output Token | 首 token 之后，平均每个输出 token 的耗时，"
      "= (最后一个内容 chunk 时刻 − 首 token 时刻) / (输出 token 数 − 1)。"
      "| 文字持续吐出的速度。20ms/token 约等于 50 token/s，快于多数人阅读速度。 |")
    A("| **ITL** | Inter-Token Latency | 相邻两个流式 chunk 的到达间隔，逐个采样。"
      "TPOT 是它的均值，ITL 保留了分布。| 卡顿。均值正常但 Max 很大说明输出一顿一顿的。 |")
    A("| **E2E** | End-to-End Latency | 单请求从发出到完全结束的总耗时。| "
      "批处理与异步任务的成本核算依据。 |")
    A("| **TPS** | Tokens Per Second | 整体吞吐，全部输出 token 数 ÷ 压测墙钟时长，"
      "跨请求累加。| 硬件产出效率，决定单位算力能服务多少业务量。 |")
    A("| **Goodput** | — | 每秒**同时满足所有 SLO** 的成功请求数。"
      "超时或太慢的请求不计入。| 真实可交付容量。见 §4 的解释。 |")
    A("")
    A("> TPS 有两个口径，容易混淆：**整体吞吐**是全部请求叠加的结果，"
      "并发越高越大；**单请求吞吐**是单个用户体感的出字速度，并发越高越小。"
      "两者不可比较，报告中分列。")
    A("")

    A("### 5.2 结果表格")
    A("")
    A("| 场景 | 并行度 | 总请求 | 成功 | 失败 | "
      "TTFT P50 (s) | TTFT P95 (s) | TTFT P99 (s) | "
      "TPOT P50 (ms) | TPOT P95 (ms) | TPOT P99 (ms) | "
      "E2E P50 (s) | E2E P95 (s) | E2E P99 (s) | 整体吞吐 (tok/s) | "
      "单请求吞吐 P50 (tok/s) | Goodput (req/s) |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in steps:
        load = s.concurrency if s.mode == "closed" else f"{s.request_rate}/s"
        A(
            f"| {s.name} | {load} | {s.total_requests} | {s.ok_requests} | "
            f"{s.failed_requests} | "
            f"{_sec(s.ttft_ms,'p50')} | {_sec(s.ttft_ms,'p95')} | "
            f"{_sec(s.ttft_ms,'p99')} | "
            f"{_cell(s.tpot_ms,'p50')} | {_cell(s.tpot_ms,'p95')} | "
            f"{_cell(s.tpot_ms,'p99')} | "
            f"{_sec(s.e2e_ms,'p50')} | {_sec(s.e2e_ms,'p95')} | "
            f"{_sec(s.e2e_ms,'p99')} | "
            f"{s.output_tps:.1f} | {_cell(s.per_request_tps,'p50')} | "
            f"{s.goodput_rps:.3f} |"
        )
    A("")
    A("延迟单位：TTFT / E2E 用秒，TPOT / ITL 用毫秒。"
      "分位数取 P50 / P95 / P99，Max 与 ITL 分布见 CSV。")
    A("")

    # 只在容量判定所用的那个长度档内分区：不同长度档的 Goodput 不可比，
    # 且并行度可能重号（如 p4 与长上下文档都是 4），混排会出现两个 C4
    b_in = saturation.get("basis_input_tokens")
    b_out = saturation.get("basis_output_tokens")
    closed = [s for s in steps if s.mode == "closed"]
    if b_in is not None:
        closed = [s for s in closed
                  if s.target_input_tokens == b_in
                  and s.target_output_tokens == b_out]
    if len(closed) >= 2:
        best = max(closed, key=lambda s: s.goodput_rps)
        A("### 5.3 工作区间分析")
        A("")
        A(f"按 Goodput 与延迟的联动关系，把并行度阶梯划成三段。"
          f"**仅针对 input={b_in} / output={b_out} 这一档**——"
          f"其他长度档的阶梯不完整或长度不可比，不参与分区"
          f"（见 §8.1）。**数据只说明现象，分区给出的是工程含义。**")
        A("")
        low = [s for s in closed
               if s.goodput_ratio >= 0.999 and s.concurrency < best.concurrency]
        over = [s for s in closed if s.goodput_ratio < 0.5
                and s.concurrency > best.concurrency]
        mid = [s for s in closed if s not in low and s not in over
               and s is not best]

        def _tag(rows: list[ScenarioStats]) -> str:
            return "、".join(f"C{r.concurrency}" for r in rows)

        if low:
            A(f"**低负载区（{_tag(low)}）**")
            A("")
            for s in low:
                A(f"- C{s.concurrency}：TTFT P95 {_sec(s.ttft_ms, 'p95')}s，"
                  f"Goodput {s.goodput_rps:.3f} req/s，达标率 "
                  f"{s.goodput_ratio * 100:.0f}%，整体吞吐 {s.output_tps:.1f} tok/s")
            A("")
            A("特征：延迟平稳，Goodput 随并发近线性增长，服务端基本不排队。"
              "此区间硬件未跑满，**继续加压是划算的**——吞吐涨、体验不掉。")
            A("")
        if mid:
            A(f"**过渡区（{_tag(mid)}）**")
            A("")
            for s in mid:
                A(f"- C{s.concurrency}：TTFT P95 {_sec(s.ttft_ms, 'p95')}s，"
                  f"Goodput {s.goodput_rps:.3f} req/s，达标率 "
                  f"{s.goodput_ratio * 100:.0f}%")
            A("")
        A(f"**最佳工作点（C{best.concurrency}）**")
        A("")
        A(f"- 最大稳定 Goodput **{best.goodput_rps:.3f} req/s**，"
          f"达标率 {best.goodput_ratio * 100:.1f}%")
        A(f"- TTFT P95 {_sec(best.ttft_ms, 'p95')}s，"
          f"阈值 {best.slo_used['ttft_ms'] / 1000:.1f}s —— 仍在 SLO 内")
        A(f"- TPOT P95 {_cell(best.tpot_ms, 'p95')}ms，"
          f"阈值 {best.slo_used['tpot_ms']:.0f}ms")
        A(f"- 整体吞吐 {best.output_tps:.1f} tok/s，"
          f"单请求吞吐 P50 {_cell(best.per_request_tps, 'p50')} tok/s")
        A("")
        A("这是**推荐运行点**：再往上加并发，Goodput 不再增长甚至下降，"
          "多出来的吞吐是靠牺牲每个用户的等待时间换来的。")
        A("")
        if over:
            A(f"**过载区（{_tag(over)}）**")
            A("")
            for s in over:
                wait = (s.server_metrics or {}).get("peak_num_requests_waiting")
                w = f"，服务端排队峰值 {float(wait):.0f}" if wait else ""
                A(f"- C{s.concurrency}：整体吞吐 {s.output_tps:.1f} tok/s"
                  f"（仍在涨），但 TTFT P95 {_sec(s.ttft_ms, 'p95')}s、"
                  f"P99 {_sec(s.ttft_ms, 'p99')}s，"
                  f"Goodput 掉到 {s.goodput_rps:.3f} req/s"
                  f"（达标率 {s.goodput_ratio * 100:.1f}%）"
                  f"，单请求吞吐 P50 {_cell(s.per_request_tps, 'p50')} tok/s{w}")
            A("")
            A("特征与成因：并发超过服务端单批容量（`max_num_seqs` / KV cache 槽位）后，"
              "多出的请求进 waiting 队列等调度，这段排队**全部计入 TTFT**。"
              "此时 GPU/NPU 始终满负荷，所以整体 TPS 继续上升；"
              "但 TPOT 往往仍然正常——**一旦轮到就算得很快，问题全在轮不到**。"
              "用户侧的体感是「点了半天没反应，一开始出字就很流畅」。")
            A("")
            A("> 这一段是全篇最容易误读的地方：**TPS 最高的那一档，是体验最差的一档。**"
              "以 TPS 选容量会直接选到过载区。")
            A("")

    A("## 6. SLO 达标与 Goodput 归因")
    A("")
    A("**成功 ≠ 达标。** HTTP 200 且输出完整只说明请求没出错，"
      "不说明它够快。Goodput 只统计**同时满足全部 SLO 阈值**的请求，"
      "所以完全可能出现「成功率 100%、Goodput 占比 0%」——"
      "所有请求都返回了，但没有一个在用户能接受的时间内返回。"
      "这正是 §1.1 把 Goodput 排在 TPS 之前的原因。")
    A("")
    A("下表把每档的违约请求按原因拆开。同一请求可能同时违反多项，"
      "故各列之和可能大于请求数。")
    A("")
    A("| 场景 | SLO 阈值 TTFT/TPOT (ms) | 成功 | 失败 | "
      "达标 (Goodput) | 占比 | TTFT 超标 | TPOT 超标 | Stall 率 | 错误分布 |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for s in steps:
        errs = ", ".join(f"{k}={v}" for k, v in s.error_breakdown.items()) or "无"
        v = s.slo_violations
        n_good = round(s.goodput_ratio * s.ok_requests)
        A(
            f"| {s.name} | {s.slo_used['ttft_ms']:.0f} / {s.slo_used['tpot_ms']:.0f} | "
            f"{s.ok_requests} | {s.failed_requests} | {n_good} | "
            f"{s.goodput_ratio * 100:.1f}% | {v.get('ttft', 0)} | "
            f"{v.get('tpot', 0)} | {s.stall_rate * 100:.1f}% | {errs} |"
        )
    A("")

    # 逐档给出可读的归因结论，避免读者自己比对数字
    zero = [s for s in steps if s.ok_requests > 0 and s.goodput_ratio < 0.999]
    if zero:
        A("### 6.1 逐档归因")
        A("")
        for s in zero:
            v = s.slo_violations
            t95 = s.ttft_ms.get("p95")
            p95 = s.tpot_ms.get("p95")
            th_t = s.slo_used["ttft_ms"]
            th_p = s.slo_used["tpot_ms"]
            bits = []
            if v.get("ttft"):
                r = f"{t95 / th_t:.1f}×" if t95 else "?"
                bits.append(
                    f"**TTFT 超标 {v['ttft']}/{s.ok_requests} 条**"
                    f"（P95 {_fmt(t95)}ms，阈值 {th_t:.0f}ms，{r}）"
                )
            if v.get("tpot"):
                r = f"{p95 / th_p:.1f}×" if p95 else "?"
                bits.append(
                    f"**TPOT 超标 {v['tpot']}/{s.ok_requests} 条**"
                    f"（P95 {_fmt(p95)}ms，阈值 {th_p:.0f}ms，{r}）"
                )
            if v.get("e2e"):
                bits.append(f"E2E 超标 {v['e2e']} 条")
            if v.get("failed"):
                bits.append(f"请求失败 {v['failed']} 条")
            if not bits:
                continue
            wait = (s.server_metrics or {}).get("peak_num_requests_waiting")
            tail = ""
            if wait:
                tail = (f" 服务端排队峰值 {float(wait):.0f} —— 请求进了队列等调度，"
                        f"这段等待全部计入 TTFT。")
            A(f"- **{s.name}**（并行度 {s.concurrency}）："
              f"{'；'.join(bits)}。{tail}")
        A("")
        A("排队是 TTFT 崩掉的常见主因：并发数超过服务端单批能容纳的请求数后，"
          "多出来的请求在 waiting 队列里等前一批腾出 KV cache 槽位。"
          "此时整体 TPS 仍可能上升（GPU/NPU 一直在满负荷算），"
          "但每个用户的首字等待被拉长数倍。**只看 TPS 会选出体验最差的工作点。**")
        A("")

    server_rows = [s for s in steps if (s.server_metrics or {}).get("available")]
    if server_rows:
        A("## 7. 服务端指标交叉校验")
        A("")
        A("客户端 TTFT 含网络往返，服务端 histogram 不含。差值超过 10% "
          "说明测试机或链路是瓶颈，该组数据不代表服务能力。")
        A("")
        A("| 场景 | 客户端 TTFT P95 | 服务端 TTFT P95 | 差值 | "
          "服务端 QueueTime P95 | Waiting 峰值 | KV 峰值 | Prefix 命中率 |")
        A("|---|---|---|---|---|---|---|---|")
        has_bound = False
        for s in server_rows:
            sm = s.server_metrics or {}
            c95 = s.ttft_ms.get("p95")
            s95 = _srv(sm, "server_time_to_first_token_p95_ms")
            raw95 = sm.get("server_time_to_first_token_p95_ms")
            bounded = sm.get("server_time_to_first_token_p95_upper_bound_only")
            # 上界值不能参与差值计算，否则得出的偏差没有意义
            gap = (
                f"{(c95 - raw95) / raw95 * 100:+.1f}%"
                if c95 and raw95 and not bounded else "-"
            )
            q95 = _srv(sm, "server_request_queue_time_p95_ms")
            if "<=" in s95 or "<=" in q95:
                has_bound = True
            hit = sm.get("prefix_cache_hit_rate")
            A(
                f"| {s.name} | {_fmt(c95)} | {s95} | {gap} | {q95} | "
                f"{_fmt(sm.get('peak_num_requests_waiting'), 0)} | "
                f"{_fmt(sm.get('peak_kv_cache_usage_perc'), 3)} | "
                f"{f'{hit * 100:.1f}%' if hit is not None else '-'} |"
            )
        A("")
        if has_bound:
            A("> `<= X` 表示该分位数落在 histogram 的首个分桶内，桶内分布未知，"
              "只能给出上界，不是测量值。vLLM 的 queue_time / prefill_time "
              "最低分桶为 300ms，小于此值的真实延迟无法从 `/metrics` 区分。")
            A("")

        A("### 7.1 诊断逻辑")
        A("")
        A("交叉校验的用途是**定位瓶颈在哪一侧**，按下表对号入座：")
        A("")
        A("| 观察到的现象 | 判断 | 下一步 |")
        A("|---|---|---|")
        A("| 客户端 TTFT ≫ 服务端 TTFT，且服务端 QueueTime 低 | "
          "瓶颈在服务端之外：网络往返、SDK 自身开销、或压测端 CPU/事件循环饱和 | "
          "在测试机本地 `curl` 测裸往返；检查压测进程 CPU 是否打满；"
          "确认差值不是 histogram 分桶假象（服务端值带 `<=` 时不可用于比较） |")
        A("| 客户端与服务端 TTFT 接近（差值 < 10%） | 客户端数据可信，"
          "反映真实服务能力 | 正常采信 |")
        A("| 服务端 QueueTime 显著升高 | 请求在 waiting 队列等调度："
          "并发已超单批容量，或 KV cache 槽位不足 | "
          "查 Waiting 峰值与 KV 峰值：KV 接近 1.0 说明显存受限，"
          "调 `gpu_memory_utilization` 或降 `max_model_len`；"
          "KV 不高但 Waiting 高说明受 `max_num_seqs` 限制 |")
        A("| QueueTime 低但 TPOT 高 | 排队正常，是解码本身慢 | "
          "查 batch 内 token 竞争、算子实现、`enforce_eager` 是否误开 |")
        A("| Prefix 命中率异常高 | prompt 复用导致 TTFT 偏乐观 | "
          "确认 §4 的 prompt 策略为 `unique`，否则结论不代表冷启动能力 |")
        A("")
        A("> 客户端 ≫ 服务端时**不要直接下结论说链路有问题**："
          "先确认服务端分位数是否为分桶上界估算。"
          "低延迟档（真实值 < 300ms）两侧差值大属正常现象，不构成链路故障证据。")
        A("")

    A("## 8. 容量判定")
    A("")
    if not saturation.get("saturated") and saturation.get("reason"):
        A(f"未做饱和点判定：{saturation['reason']}（需至少两个并发阶梯）。")
    else:
        A("**两个概念必须分开，生产上用途不同：**")
        A("")
        A("- **服务饱和点**：系统开始无法满足 SLO 的位置。这是**上限告警线**，"
          "不是运行目标——跑在这里意味着已有用户体验不达标。")
        A("- **推荐运行点**：最大 Goodput 对应的并发。这是**容量规划依据**，"
          "即在体验达标前提下能交付的最大请求量。")
        A("")
        A("推荐运行点通常低于饱和点。两者相等说明阶梯粒度太粗，"
          "或最优点落在两档之间，需加密并发档位重测。")
        A("")
        A("### 8.1 判定依据的长度档")
        A("")
        A("容量判定只在**相同输入/输出长度**的并行度阶梯内进行。"
          "长度不同的场景吞吐本就不可比——长输入 prefill 更重、"
          "长输出摊薄 TTFT 占比——混在一起会把「换长度档导致的吞吐下降」"
          "误判成「并行度到顶」。")
        A("")
        A("| 项 | 值 |")
        A("|---|---|")
        A(f"| 判定依据 input_tokens | {saturation.get('basis_input_tokens', '-')} |")
        A(f"| 判定依据 output_tokens | {saturation.get('basis_output_tokens', '-')} |")
        A(f"| 参与判定的并行度 | "
          f"{saturation.get('basis_parallelism_steps', '-')} |")
        excl = saturation.get("excluded_groups") or []
        if excl:
            txt = "；".join(
                f"in={g['input_tokens']}/out={g['output_tokens']}"
                f"(并行度 {g['parallelism']})" for g in excl
            )
            A(f"| 未参与判定的长度档 | {txt} |")
        A("")
        if excl:
            A("> 上列长度档未参与容量判定，其结果仅用于观察长度对延迟的影响"
              "（见 §5.2）。要得到这些档位的容量结论，需在该长度下单独跑"
              "并行度阶梯。")
            A("")
        A("### 8.2 容量结论")
        A("")
        A("| 指标 | 结果 |")
        A("|---|---|")
        A(f"| 最大稳定 Goodput | **{_fmt(saturation.get('max_goodput_rps'), 3)} req/s** |")
        A(f"| **推荐运行并发** | **C{_fmt(saturation.get('max_goodput_at_concurrency'), 0)}** |")
        A(f"| **服务饱和点** | **C{_fmt(saturation.get('saturation_concurrency'), 0)}** |")
        A(f"| 建议限流值（Goodput × 80%） | "
          f"{_fmt(saturation.get('recommended_rate_limit_rps'), 3)} req/s |")
        A("")
        A("### 8.3 判据明细")
        A("")
        A("饱和点取三条判据中**最早触发**的那一档：")
        A("")
        A("| 判据 | 触发位置 | 含义 |")
        A("|---|---|---|")
        A(f"| Output TPS 停止上升 | 并发 {_fmt(saturation.get('tps_plateau_at'), 0)} | "
          "硬件算力跑满，加压不再增加产出 |")
        A(f"| Goodput 开始下降 | 并发 {_fmt(saturation.get('goodput_decline_at'), 0)} | "
          "可交付请求数减少，加压反而有害 |")
        A(f"| P95 越过 SLO | 并发 {_fmt(saturation.get('slo_violation_at'), 0)} | "
          "体验不达标，用户已能感知 |")
        A("")
        A("> 某条判据显示 `-` 表示在本次测试的并发范围内未触发。"
          "TPS 平台期未触发说明尚未压到算力上限，"
          "此时饱和点由 SLO 判据给出，是**体验上限而非算力上限**——"
          "要定位算力上限需继续加大并发档位。")
        A("")
        A(f"> {saturation.get('recommended_rate_limit_note', '')}")
    A("")

    A("## 9. 硬件资源监控")
    A("")
    hw = env.get("hardware") or {}
    A("推理性能与硬件占用必须成对解读：**TPS 不再上升时，"
      "要靠利用率区分「算力跑满」还是「被别处卡住」**。"
      "利用率低而延迟高，说明瓶颈在调度、显存或通信，不是算力不足。")
    A("")
    A("### 9.1 本轮采集状态")
    A("")
    A("| 项 | 值 |")
    A("|---|---|")
    A(f"| 加速器类型 | {_tbd(hw, 'accelerator')} |")
    A(f"| 卡数 | {_tbd(hw, 'device_count')} |")
    A(f"| 采集方式 | {_tbd(hw, 'collector')} |")
    A(f"| 采集间隔 | {_tbd(hw, 'sample_interval_s')} |")
    A("")
    if not hw:
        A("> **本轮未采集硬件指标。** 当前为 SDK 流程验证阶段，"
          "被测服务运行在测试机上，硬件占用不具备生产参考价值，故未接入采集。"
          "指标值一律记为 `[TBD]`，**不以任何方式推断或估算**。")
        A("")
    A("### 9.2 采集项定义")
    A("")
    A("| 类别 | NVIDIA（`nvidia-smi` / DCGM） | Ascend（`npu-smi info`） | 本轮值 |")
    A("|---|---|---|---|")
    A(f"| 计算利用率 | `utilization.gpu` | AICore Usage | {_tbd(hw, 'util')} |")
    A(f"| 显存 / HBM 占用 | `memory.used` / `memory.total` | HBM-Usage | "
      f"{_tbd(hw, 'memory')} |")
    A(f"| 功耗 | `power.draw` | Power | {_tbd(hw, 'power')} |")
    A(f"| 温度 | `temperature.gpu` | Temp | {_tbd(hw, 'temp')} |")
    A(f"| 频率 | `clocks.sm` | AICore Freq | {_tbd(hw, 'clock')} |")
    A(f"| 通信状态 | NVLink / PCIe 带宽 | **HCCL 链路状态与带宽** | "
      f"{_tbd(hw, 'interconnect')} |")
    A("")
    A("### 9.3 多卡部署的强制采集要求")
    A("")
    A("**跨卡（TP / PP / EP）部署下必须采集上述全部项**，"
      "缺失则容量结论不可采信。理由：")
    A("")
    A("- **显存 / HBM 占用**：权重占满后留给 KV cache 的余量直接决定可并发数。"
      "不采集则无法判断「排队」是受 `max_num_seqs` 限制还是显存耗尽——"
      "两者的扩容手段完全不同。")
    A("- **互联通信**：跨卡切分下通信耗时计入每个 decode step，"
      "表现为 TPOT 升高而计算利用率不高。"
      "无通信数据时这种情况会被误判为算力不足，进而错误地采购更多算力。")
    A("- **功耗与温度**：长时间稳定性测试中降频会导致 TPOT 缓慢漂移，"
      "只有配合温度曲线才能识别，否则会误判为内存泄漏或负载不均。")
    A("- **计算利用率**：区分 §5.3 过载区是「算力饱和」还是「调度饱和」的唯一依据。")
    A("")
    A("采集方式建议：压测期间以固定间隔轮询 `nvidia-smi` / `npu-smi info`，"
      "按时间戳与请求记录对齐，输出与延迟曲线同轴的资源曲线。"
      "**采样间隔不应大于 5s**，否则会漏掉 prefill 阶段的瞬时峰值。")
    A("")

    A("## 10. 后续测试计划")
    A("")
    A("当前仅完成 SDK 流程验证与单节点容量扫描。生产级评测需补齐下列维度，"
      "均为**未执行**项。")
    A("")
    A("| 阶段 | 内容 | 目的 | 状态 |")
    A("|---|---|---|---|")
    A("| 基础性能 | 单请求 baseline（并发 1，无竞争） | "
      "取无排队干扰的纯推理延迟，作为所有并发档的对照基准 | [TBD] |")
    A("| 容量扫描 | C1 / C4 / C8 / C16 / C32 / C48 | "
      "定位推荐运行点与饱和点；档位加密以避免最优点落在两档之间 | 部分完成 |")
    A("| 长上下文 | 1K / 8K / 32K 输入 | "
      "prefill 随输入长度超线性增长，长上下文下 TTFT 与 KV cache "
      "占用的表现与短输入完全不同 | [TBD] |")
    A("| 长时间稳定性 | 30min / 60min 恒定负载 | "
      "识别显存碎片、缓存膨胀、降频导致的性能漂移——"
      "短时压测无法暴露 | [TBD] |")
    A("| 开环到达 | 泊松到达（固定 QPS 而非固定并发） | "
      "闭环负载会自我限速，无法反映真实流量突发下的排队行为 | [TBD] |")
    A("| 多节点 / 并行策略 | TP、PP、EP 与互联拓扑对比 | "
      "确定被测模型在目标硬件上的最优切分方式；"
      "MoE 模型的 EP 配置对吞吐影响显著 | [TBD] |")
    A("")

    A("## 11. 数据说明")
    A("")
    A("- 延迟分位数由客户端逐请求测量后线性插值得出；服务端分位数由 "
      "`/metrics` histogram 分桶估算，两者方法不同，不可直接比较绝对值。")
    A("- TPOT 分母为 `output_tokens - 1`，首 token 耗时归入 TTFT，不重复计入。")
    A("- 预热请求不计入统计。")
    A("- `ignore_eos=true` 时输出定长，便于横向对比；关闭后输出长度由模型决定。")
    A("- Goodput 依赖 SLO 取值，SLO 变更后所有容量结论需重算。")
    A("- `[TBD]` 表示该项未采集或未声明，**不做任何推断或估算**。"
      "跨轮次、跨硬件对比前必须补齐 §3 与 §9 的 `[TBD]` 项，"
      "否则性能差异无法归因到具体变量。")
    A("- §5.3 的工作区间划分由实测 Goodput 达标率自动生成，"
      "分区阈值为达标率 99.9% 与 50%；区间内的工程解释为通用推理服务行为，"
      "不代表本轮逐档验证结论。")
    return "\n".join(L)


def save_all(out_dir: str, run_id: str, env: dict[str, Any],
             steps: list[ScenarioStats]) -> dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    closed = [s for s in steps if s.mode == "closed"]
    sat = find_saturation(closed) if len(closed) >= 2 else {
        "saturated": False, "reason": "insufficient_steps"
    }

    paths = {
        "json": os.path.join(out_dir, f"{run_id}_summary.json"),
        "markdown": os.path.join(out_dir, f"{run_id}_report.md"),
        "csv": os.path.join(out_dir, f"{run_id}_curves.csv"),
    }
    write_json(paths["json"], {
        "run_id": run_id,
        "env": env,
        "scenarios": [s.to_json() for s in steps],
        "saturation": sat,
    })
    with open(paths["markdown"], "w", encoding="utf-8") as f:
        f.write(render_markdown(env, steps, sat))
    write_curves_csv(paths["csv"], steps)
    return paths
