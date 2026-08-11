"""命令行入口。

    perfkit probe   -c conf.json
    perfkit smoke   -c conf.json
    perfkit sweep   -c conf.json --parallelism 1,4,8,16,32
    perfkit run     -c conf.json

安装后 perfkit 即为控制台命令；未安装时等价于 python -m perfkit.cli。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime

from .config import ScenarioSpec, build_scenarios, load_config
from .metrics import MetricsCollector
from .record import JsonlWriter
from .report import save_all
from .runner import ScenarioRunner, probe_target
from .tokens import TokenCounter

_EPILOG = """\
示例：
  # 探测服务能力（models / metrics / tokenize）
  perfkit probe -c conf.json

  # 冒烟：3 个请求，确认链路通
  perfkit smoke -c conf.json

  # 并行度阶梯，单一长度组合
  perfkit sweep -c conf.json \\
      --parallelism 1,4,8,16 --input-tokens 1024 --output-tokens 128 \\
      --multiplier 10

  # 三维矩阵：4 × 2 × 2 = 16 个场景
  perfkit sweep -c conf.json \\
      --parallelism 1,4,8,16 --input-tokens 1024,4096 --output-tokens 128,512

  # 执行配置文件里定义的 scenarios（支持 list 值自动展开矩阵）
  perfkit run -c conf.json

口径：
  总请求数 = parallelism × multiplier。倍率固定才能让各并行档的稳态时长
  大致相当，横向对比才成立；用固定总数会让低并行档跑很多轮、高并行档只跑
  一轮，调度器行为不可比。
"""


def _log(msg: str) -> None:
    print(msg, flush=True)


def _run_id() -> str:
    # 本地时间戳，便于与服务端日志对齐
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _env_snapshot(cfg, probe: dict, counter: TokenCounter) -> dict:
    warnings: list[str] = []
    if not cfg.target.has_metrics:
        warnings.append(
            "/metrics 不可用：无服务端交叉校验，排队与 KV 使用率缺失"
        )
    if not counter.available:
        warnings.append(
            "/tokenize 不可用：input_tokens 为字符数估算值，与服务端计费口径可能不一致"
        )
    if probe.get("model_mismatch"):
        warnings.append(probe["model_mismatch"])
    return {
        "label": cfg.target.label or "-",
        "base_url": cfg.target.base_url,
        "model": cfg.target.model,
        "max_model_len": cfg.target.max_model_len,
        "has_metrics": cfg.target.has_metrics,
        "has_tokenize": cfg.target.has_tokenize,
        "token_source": "server_/tokenize" if counter.available else "char_estimate",
        "sdk_version": "0.1.0",
        "python": sys.version.split()[0],
        "seed": cfg.seed,
        "deployment": cfg.deployment,
        "hardware": cfg.hardware,
        "probe": probe,
        "warnings": warnings,
    }


async def _execute(cfg, specs: list[ScenarioSpec], out_dir: str) -> None:
    probe = await probe_target(cfg.target)
    if not probe.get("reachable"):
        _log(f"服务不可达：{probe.get('error')}")
        sys.exit(1)

    _log(f"服务可达 | models={probe.get('models')} "
         f"max_model_len={probe.get('max_model_len')} "
         f"metrics={probe.get('has_metrics')} tokenize={probe.get('has_tokenize')}")

    errs: list[str] = []
    for s in specs:
        errs.extend(s.validate(cfg.target.max_model_len))
    if errs:
        for e in errs:
            _log(f"配置错误：{e}")
        sys.exit(2)

    os.makedirs(out_dir, exist_ok=True)
    rid = _run_id()
    counter = TokenCounter(cfg.target)
    collector = MetricsCollector(cfg.target) if cfg.target.has_metrics else None
    jsonl = os.path.join(out_dir, f"{rid}_requests.jsonl")

    steps = []
    with JsonlWriter(jsonl) as writer:
        runner = ScenarioRunner(cfg.target, counter, collector, writer,
                                progress=_log, seed=cfg.seed)
        for i, spec in enumerate(specs, 1):
            load = spec.parallelism if spec.mode == "closed" else f"{spec.request_rate}/s"
            _log(f"[{i}/{len(specs)}] {spec.name} "
                 f"mode={spec.mode} load={load} "
                 f"in={spec.input_tokens} out={spec.output_tokens} "
                 f"prefix={spec.prefix_mode}")
            steps.append(await runner.run(spec, cfg.slo))

    env = _env_snapshot(cfg, probe, counter)
    counter.close()
    if collector:
        collector.close()

    paths = save_all(out_dir, rid, env, steps)
    _log("")
    for w in env["warnings"]:
        _log(f"提示：{w}")
    _log(f"原始记录 {jsonl}")
    for k, v in paths.items():
        _log(f"{k:8s} {v}")


def _int_list(s: str, flag: str) -> list[int]:
    """解析逗号分隔的整数列表。压测跑几十分钟，参数错误要在启动瞬间就失败。"""
    try:
        vals = [int(x) for x in s.split(",") if x.strip()]
    except ValueError as e:
        raise SystemExit(f"{flag} 只接受逗号分隔的整数，收到：{s!r}") from e
    if not vals:
        raise SystemExit(f"{flag} 不能为空")
    if any(v < 1 for v in vals):
        raise SystemExit(f"{flag} 的值需 >= 1，收到：{vals}")
    return vals


def _sweep_specs(base: ScenarioSpec, pars: list[int], ins: list[int],
                 outs: list[int], multiplier: int) -> list[ScenarioSpec]:
    """按 parallelism × input × output 三维矩阵展开场景。

    总请求数 = 并行度 × 倍率，与基线表口径一致（该表倍率为 10）。
    base.num_requests 非空时按固定值执行，此时各档样本量不随并行度放大，
    低并行档的分位数置信度会低于高并行档，报告中需注明。

    复用 config.expand_matrix 而不在这里手写三重循环，是为了让 CLI 与 JSON
    两条入口的展开规则和命名后缀完全一致——否则同一组参数经两条路径跑出来的
    场景名不同，报告无法横向对比。
    """
    raw = {
        "name": base.name,
        "parallelism": pars,
        "input_tokens": ins,
        "output_tokens": outs,
        "mode": "closed",
        "num_requests": base.num_requests,
        "request_multiplier": 0 if base.num_requests else multiplier,
        "warmup_requests": base.warmup_requests,
        "prefix_mode": base.prefix_mode,
        "shared_prefix_ratio": base.shared_prefix_ratio,
        "temperature": base.temperature,
        "ignore_eos": base.ignore_eos,
        "slo": base.slo,
    }
    specs = build_scenarios([raw])
    for s in specs:
        s.resolve_requests()
    return specs


def main() -> None:
    # allow_abbrev=False：--out 会同时匹配 --output-tokens 和 --out-dir 而报错，
    # 压测跑几十分钟，参数在启动瞬间就该确定地失败或生效，不留缩写歧义。
    ap = argparse.ArgumentParser(
        prog="perfkit", allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="LLM 推理服务性能压测框架。被测对象只需 OpenAI 兼容接口。",
        epilog=_EPILOG,
    )
    ap.add_argument("cmd", choices=["probe", "smoke", "sweep", "run"],
                    help="probe 探测服务能力；smoke 3 请求冒烟；"
                         "sweep 按矩阵扫描；run 执行配置里的 scenarios")
    ap.add_argument("-c", "--config", required=True, help="JSON 配置路径")
    # 三个矩阵维度都收逗号列表，任一给多值即展开笛卡尔积
    ap.add_argument("--parallelism", "--concurrency", dest="parallelism",
                    default="1,4,8,16", metavar="P[,P...]",
                    help="并行度阶梯（同时在飞的请求数），逗号分隔。"
                         "--concurrency 为兼容旧脚本的别名")
    ap.add_argument("--input-tokens", default="1024", metavar="N[,N...]",
                    help="目标输入长度（token），可给多值扫描。"
                         "实际长度按 /tokenize 逼近，近似即可")
    ap.add_argument("--output-tokens", default="128", metavar="N[,N...]",
                    help="目标输出长度（token），可给多值扫描。"
                         "配 ignore_eos 时为精确值")
    ap.add_argument("--multiplier", "--task-multiplier", dest="multiplier",
                    type=int, default=10,
                    help="并行任务倍率：一个并行位跑多少请求，"
                         "总请求数 = 并行度 × 该值（基线表口径为 10）")
    ap.add_argument("--requests", type=int, default=0,
                    help="各档固定请求数，覆盖 --multiplier。0 表示用倍率折算")
    ap.add_argument("-o", "--out-dir", default="", help="输出目录，缺省取配置值")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = args.out_dir or cfg.out_dir

    if args.cmd == "probe":
        info = asyncio.run(probe_target(cfg.target))
        print(json.dumps(info, ensure_ascii=False, indent=2))
        sys.exit(0 if info.get("reachable") else 1)

    if args.cmd == "smoke":
        specs = [ScenarioSpec(
            name="smoke", input_tokens=128, output_tokens=32,
            mode="closed", parallelism=1, num_requests=3,
            warmup_requests=1, prefix_mode="unique",
        )]
    elif args.cmd == "sweep":
        pars = _int_list(args.parallelism, "--parallelism")
        ins = _int_list(args.input_tokens, "--input-tokens")
        outs = _int_list(args.output_tokens, "--output-tokens")
        base = ScenarioSpec(
            name="sweep", input_tokens=ins[0], output_tokens=outs[0],
            mode="closed", num_requests=args.requests or None,
            warmup_requests=2, prefix_mode="unique",
        )
        specs = _sweep_specs(base, pars, ins, outs, args.multiplier)
        n_dim = len(pars) * len(ins) * len(outs)
        _log(f"测试矩阵 parallelism={pars} × input={ins} × output={outs} "
             f"= {n_dim} 个场景")
        if args.requests:
            _log(f"注意：--requests {args.requests} 覆盖倍率，各档样本量相同"
                 f"（共 {args.requests * n_dim} 请求）")
        else:
            tot = sum(s.num_requests or 0 for s in specs)
            _log(f"并行任务倍率 {args.multiplier}，"
                 f"各档请求数 = 并行度 × 倍率，共 {tot} 请求")
    else:
        specs = cfg.scenarios
        if not specs:
            _log("配置里没有 scenarios")
            sys.exit(2)
        for s in specs:
            s.resolve_requests()

    asyncio.run(_execute(cfg, specs, out_dir))


if __name__ == "__main__":
    main()
