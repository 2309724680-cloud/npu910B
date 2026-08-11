# llm-perfkit

面向 OpenAI 兼容推理服务的性能压测工具。

只要目标暴露 `/v1/chat/completions`——vLLM、SGLang、TGI、llama.cpp server 或托管
API 都算——就能测 **TTFT、TPOT、ITL、吞吐与 Goodput**，支持闭环与开环两种负载
模型，可按并行度与序列长度扫描矩阵，采集服务端 vLLM `/metrics`，输出
Markdown + CSV + JSON 报告。换被测对象只改配置。

已在 NVIDIA GPU 与昇腾 NPU 上的 vLLM 实测验证。

English: [README.md](README.md)

## 为什么不用平均延迟就够了

多数压测工具只报一个平均延迟和一个吞吐值，而这两个数恰好掩盖了要做的决策：

- **吞吐有两个方向相反的口径。** 整体 Output TPS 随并行度上升，单请求
  Generation TPS 随并行度下降。报错口径就会高估或低估集群容量。
- **最大吞吐点通常不能用作生产工作点。** 吞吐最高的并行度往往已经越过 P95 TTFT
  的 SLO 线。工具因此报 **Goodput**——只计入满足 SLO 的请求的吞吐——推荐工作点
  由数据直接给出。
- **分位数比均值重要。** 一次把整段响应冲出来的流会产出无意义的单请求 TPS。
  工具同时给中位数与 P95/P99，让异常暴露出来而不是被均值吸收。

指标定义、测量口径与各指标的已知局限见 [docs/methodology.md](docs/methodology.md)。

## 安装

需要 Python 3.10+。

```bash
pip install llm-perfkit
```

从源码：

```bash
git clone https://github.com/2309724680-cloud/npu910B.git
cd npu910B
pip install -e .
```

可选依赖：

```bash
pip install "llm-perfkit[charts]"   # matplotlib，画曲线
pip install "llm-perfkit[xlsx]"     # openpyxl，导出表格
pip install -e ".[dev]"             # pytest + ruff
```

唯一硬依赖是 `httpx`。不需要 tokenizer、CUDA 或厂商 SDK——token 数优先取服务端
`/tokenize`，没有该接口时退化为字符估算，并在报告中标明来源。

## 快速开始

复制示例配置，改掉端点：

```bash
cp examples/quickstart.json conf.json
export LLM_API_KEY=...        # 服务端无鉴权时可省略
```

```json
{
  "target": {
    "base_url": "http://YOUR_SERVER_HOST:8000/v1",
    "model": "YOUR_MODEL_NAME",
    "api_key": "${LLM_API_KEY}"
  }
}
```

`api_key` 支持 `${ENV_VAR}` 展开，密钥不必入库。

然后按三步推进：

```bash
# 1. 探测：是否可达、max_model_len 多少、/metrics 与 /tokenize 是否可用
perfkit probe -c conf.json

# 2. 冒烟：3 个请求，确认流式链路通
perfkit smoke -c conf.json

# 3. 阶梯：并行度扫描，每个并行位 10 个请求
perfkit sweep -c conf.json --parallelism 1,4,8,16,32 \
    --input-tokens 1024 --output-tokens 128 --multiplier 10
```

三个维度可任意组合，按笛卡尔积展开，下面是 `4 × 2 × 2 = 16` 个场景：

```bash
perfkit sweep -c conf.json --parallelism 1,4,8,16 \
    --input-tokens 1024,4096 --output-tokens 128,512
```

也可以把场景写进配置批量执行（list 值同样自动展开，见
[examples/concurrency-sweep.json](examples/concurrency-sweep.json)）：

```bash
perfkit run -c conf.json
```

产物落在 `out_dir`（默认 `results/`）：

| 文件 | 内容 |
| --- | --- |
| `<run>_requests.jsonl` | 每请求一条原始记录，可重新分析 |
| `<run>_report.md` | Markdown 报告：分场景表格、饱和点分析、口径提醒 |
| `<run>_curves.csv` | 每场景一行，四条标准曲线，可直接进表格 |
| `<run>_summary.json` | 同样的汇总，机器可读 |

`results/` 已进 `.gitignore`——压测产物描述的是你的基础设施，不该提交。

## 测什么

| 指标 | 定义 |
| --- | --- |
| **TTFT** | 首 token 时延，逐请求 |
| **TPOT** | 首 token 之后的平均每 token 时延 |
| **ITL** | token 间隔，保留完整分布 |
| **Output TPS** | 整体输出吞吐，按 wall-clock 计——随并行度上升 |
| **Generation TPS** | 单请求输出吞吐——随并行度下降 |
| **Goodput** | 同时满足全部 SLO 的请求速率（成功 ∧ TTFT ≤ SLO ∧ TPOT ≤ SLO） |
| **Stall Rate** | 出现超过 `stall_itl_ms` 的 ITL 间隔的请求占比 |
| 饱和点 | 继续加并行度不再换来吞吐的位置 |

负载模型：**闭环**（固定并行度，N 个在飞）与**开环**（泊松到达，给定速率）。
前缀控制（`unique` / `shared` / `mixed`）用于分别测有无 prefix cache 命中。

目标暴露 vLLM `/metrics` 时，会采集服务端排队深度与 KV cache 使用率，与客户端
口径交叉校验。

## 配置要点

完整字段见 [src/perfkit/config.py](src/perfkit/config.py)：

- `target.base_url` / `model` / `api_key`——换目标只需改这三项
- `slo.ttft_ms` / `tpot_ms` / `e2e_ms` / `stall_itl_ms`——Goodput 准入线，应按
  长度档位分别设定，不要用单一全局阈值
- `scenarios[].mode`——`closed` 或 `open`
- `scenarios[].parallelism`——并行度，给 list 即展开成阶梯
- `scenarios[].request_multiplier`——每个并行位的请求数（总数 =
  `parallelism × 倍率`）。倍率固定才能让样本量随并行度等比放大，各档分位数
  才可横向比较
- `scenarios[].ignore_eos`——强制定长输出，便于横向对比
- `deployment` / `hardware`——自由格式的环境快照，原样记入报告；服务端 API 不
  暴露启动参数，未声明的字段在报告中显示为 `[TBD]`

必须流式：`stream: false` 会在校验阶段直接报错而不是静默接受——没有逐 chunk
计时，TTFT / ITL / TPOT 无定义。

## 开发

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

## 许可

Apache License 2.0，见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。
