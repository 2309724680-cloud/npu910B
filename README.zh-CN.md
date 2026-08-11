# llm-perfkit

面向 OpenAI 兼容推理服务的性能压测工具。

只要目标暴露 `/v1/chat/completions`——vLLM、SGLang、TGI、llama.cpp server 或托管
API 都算——就能测 **TTFT、TPOT、ITL、吞吐与 Goodput**，支持闭环与开环两种负载
模型，可按并行度与序列长度扫描矩阵，采集服务端 vLLM `/metrics`，输出
Markdown + CSV + JSON 报告。换被测对象只改配置。

设计上与厂商无关——只对 OpenAI 兼容接口发 HTTP 请求，压测机上不需要 tokenizer、
加速卡 SDK 或任何厂商运行时。已在 **Ascend 910B × 8** 的
**DeepSeek-V4-Flash（W8A8 + MTP）** 生产部署上完成端到端实测，
详见[实测部署环境](#实测部署环境)。

[English](README.md) · **中文**

## 为什么不用平均延迟就够了

多数压测工具只报一个平均延迟和一个吞吐值，而这两个数恰好掩盖了要做的决策：

- **均值掩盖长尾。** P50 TTFT 400ms、P99 12s 是一个纸面上很好看的糟糕部署。这里
  所有延迟指标都按 p50 / p90 / p95 / p99 报，绝不单独给均值。
- **最大吞吐不等于可用吞吐。** 并行度往上推，总 tokens/s 在单请求延迟早已不可
  接受之后还会继续涨。**Goodput**——只计入满足 SLO 的请求的吞吐——才是决定生产
  工作点的那个数。
- **吞吐有两个方向相反的口径。** 整体 Output TPS 随并行度上升，单请求
  Generation TPS 随并行度下降。把一个当成另一个报出去，是这些数字最常见的误读
  方式，所以两个口径始终并列给出。

指标定义、测量口径与各指标的已知局限见 [docs/methodology.md](docs/methodology.md)。

## 安装

需要 Python 3.10+。唯一硬依赖是 `httpx`。

```bash
pip install llm-perfkit
```

可选依赖：

```bash
pip install "llm-perfkit[charts]"   # matplotlib，画曲线
pip install -e ".[dev]"             # pytest + ruff
```

从源码：

```bash
git clone https://github.com/2309724680-cloud/npu910B.git
cd npu910B
pip install -e ".[dev]"
```

不需要 tokenizer、CUDA 或厂商 SDK——token 数优先取服务端 `/tokenize`，没有该接口
时退化为字符估算，并在报告中标明来源。

## 快速开始

**1. 写配置。** 复制 [examples/quickstart.json](examples/quickstart.json)，填上
自己的端点：

```json
{
  "target": {
    "base_url": "http://YOUR_SERVER_HOST:8000/v1",
    "model": "YOUR_MODEL_NAME",
    "api_key": "${LLM_API_KEY}"
  },
  "out_dir": "results",
  "slo": { "ttft_ms": 2000, "tpot_ms": 100 }
}
```

`api_key` 写成 `${VAR}` 形式时从环境变量读取，密钥不会落盘。服务端无鉴权时留空
即可。

**2. 探测服务是否可达**，并看它暴露了什么：

```bash
perfkit probe -c conf.json
```

这一步报出模型列表、`max_model_len`，以及 `/metrics` 与 `/tokenize` 是否可用。
两个接口都是可选的，但会影响能测到什么：没有 `/tokenize` 时输入长度按字符估算，
没有 `/metrics` 时排队深度与 KV 占用就没有服务端侧的交叉校验。

**3. 冒烟测试**——3 个请求，在开始长跑之前确认整条链路是通的：

```bash
perfkit smoke -c conf.json
```

**4. 并行度阶梯：**

```bash
perfkit sweep -c conf.json --parallelism 1,4,8,16,32 \
    --input-tokens 1024 --output-tokens 128 --multiplier 10
```

每档请求总数 = `parallelism × multiplier`。这里固定的是倍率而不是总数，好让每
一档的样本量与其并行度等比放大——如果固定总数，低并行度档会跑很多轮、高并行度
档只跑一轮，两者的分位数就没法比较。

三个维度都接受 list，所以下面是一个 4 × 2 × 2 = 16 个场景的矩阵：

```bash
perfkit sweep -c conf.json --parallelism 1,4,8,16 \
    --input-tokens 1024,4096 --output-tokens 128,512
```

**5. 也可以把场景写进配置批量执行**——配置里的 list 值同样按矩阵展开（见
[examples/concurrency-sweep.json](examples/concurrency-sweep.json)）：

```bash
perfkit run -c conf.json
```

### 产物

每次运行都写入 `out_dir`，文件名带 run id 前缀：

| 文件 | 内容 |
| --- | --- |
| `<run>_requests.jsonl` | 每请求一行，完成即写 |
| `<run>_report.md` | 完整报告：分场景表格、饱和点分析、口径提醒 |
| `<run>_summary.csv` | 每场景一行，可直接进表格 |
| `<run>_curves.csv` | 四条标准曲线的整齐数据 |
| `<run>_summary.json` | 同样的汇总，机器可读 |

JSONL 边完成边落盘、逐条 flush，压测中断也留下可用数据——无需重跑即可重新汇总。

装了 `charts` 依赖后可以直接出图：

```bash
python scripts/plot_curves.py results/<run>_curves.csv -o charts
```

`results/` 已进 `.gitignore`——压测产物描述的是你自己的基础设施，不该提交。

### 作为库调用

```python
import asyncio
from perfkit import ScenarioRunner, TokenCounter, load_config, aggregate

cfg = load_config("conf.json")
counter = TokenCounter(cfg.target)
runner = ScenarioRunner(cfg.target, counter, None, None, seed=cfg.seed)

step = asyncio.run(runner.run(cfg.scenarios[0], cfg.slo))
print(step.stats.ttft_ms["p95"], step.stats.goodput_rps)
```

## 测什么

| 指标 | 含义 |
| --- | --- |
| **TTFT** | 首 token 时延——prefill + 排队，用户真正等待的那一段 |
| **TPOT** | 每输出 token 时延，在 decode 阶段上取平均 |
| **ITL** | token 间隔，TPOT 背后的原始逐 chunk 间隙 |
| **Output TPS** | 整体输出 tokens/s，按 wall-clock 计 |
| **Generation TPS** | 单请求 decode 速度，单个用户感受到的快慢 |
| **Goodput** | 成功**且**满足 TTFT 与 TPOT SLO 的请求速率 |
| **Stall rate** | 超过停顿阈值的 ITL 间隙占比 |

负载模型可以是**闭环**（固定数量的请求在飞）或**开环**（按目标速率泊松到达）。
prefix cache 按场景显式声明——`unique`、`shared` 或 `mixed`——因为共享前缀能把
TTFT 改变一个数量级，留给默认值会让结果不可比。

被测对象是 vLLM 时，每个场景前、中、后都会抓取服务端 `/metrics`，让客户端观测到
的延迟能与真实的排队深度、KV cache 占用交叉校验。这是报告能够区分「模型慢」和
「请求在排队」的依据。

## 读数注意

引用数字之前有两个坑：

- **单请求 Generation TPS 要看中位数。** 一次性把全部 token 冲出来的流会产生离谱
  的单请求速率，把均值拉偏很远。报告里两个都给，中位数才是诚实的那个。
- **`/metrics` 的 gauge 峰值是自服务启动以来的累计极值。** 用首尾差算出的排队
  深度峰值不是本场景的峰值，除非按时间窗重新采样计算。

这两点也写进了生成的报告里，没跑过压测的读者同样看得到。

## 配置要点

完整字段见 [src/perfkit/config.py](src/perfkit/config.py)：

- `target.base_url` / `model` / `api_key`——换目标只需改这三项
- `slo.ttft_ms` / `tpot_ms` / `e2e_ms` / `stall_itl_ms`——Goodput 准入线，应按
  长度档位分别设定，不要用单一全局阈值
- `scenarios[].mode`——`closed` 或 `open`
- `scenarios[].parallelism`——并行度，给 list 即展开成阶梯
- `scenarios[].request_multiplier`——每个并行位的请求数（总数 =
  `parallelism × 倍率`）
- `scenarios[].ignore_eos`——强制定长输出，便于横向对比
- `deployment` / `hardware`——自由格式的环境快照，原样记入报告；服务端 API 不
  暴露启动参数，未声明的字段在报告中显示为 `[TBD]`

必须流式：`stream: false` 会在校验阶段直接报错而不是静默降级——没有逐 chunk
计时，TTFT / ITL / TPOT 无定义。

## 实测部署环境

工具本身与硬件无关，只发 HTTP 请求。完成端到端实测的被测部署为单机昇腾环境：

| | |
|---|---|
| 模型 | DeepSeek-V4-Flash，W8A8 量化（ascend modelslim，`W8A8_DYNAMIC`） |
| 架构 | MoE，43 层，hidden 4096；routed experts 256 + shared 1，每 token 激活 top-6；MLA（64 attn heads / 1 kv head）；vocab 129,280 |
| 推测解码 | MTP，`num_speculative_tokens=1` |
| 加速卡 | Ascend 910B4-1 × 8，64 GB HBM/卡，单机 |
| 系统 / 底座 | openEuler 22.03 LTS SP3 aarch64，Ascend HDK 25.5.1，CANN 9.0.1 |
| 推理框架 | vLLM-Ascend v0.23.0rc1（vllm 0.23.0 / torch 2.10.0 / torch_npu 2.10.0.post2 / Python 3.12.13） |
| 并行策略 | TP=8 + 专家并行（EP） |
| 关键启动参数 | `--max-model-len 133120`、`--max-num-batched-tokens 8192`、`--max-num-seqs 32`、`--gpu-memory-utilization 0.9`、`--async-scheduling`、`cudagraph_mode: FULL_DECODE_ONLY`，关闭 prefix caching |
| KV 池 | 363,172 token |

已在该部署上跑通的覆盖面：并行度阶梯 1 / 2 / 4 / 8 / 12 / 16 / 20 / 24 / 28 / 32
（输入 1K、输出 128）、长度扫描（输入 8K / 32K / 64K）、以及最大上下文验证
（输入 131,072 token）。全部为闭环模式、`ignore_eos=true`、逐请求独立随机
prompt，并全程采集服务端 `/metrics`。

两点必须说明，因为它们界定了这些结果能证明什么：

- **prefix caching 是关闭的**，所以 TTFT 是冷 cache 上限。生产若有可复用系统
  提示词，实际会更优。
- **未在 NVIDIA 部署上实测。** 客户端没有任何 CUDA 相关实现，报告模板里带了一张
  `nvidia-smi` ↔ `npu-smi` 采集项对照表方便那一侧的读者，但那是换算参照，
  不是实测结论。

## 开发

```bash
pip install -e ".[dev]"
ruff check .
pytest
```

## 许可

Apache License 2.0，见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。
