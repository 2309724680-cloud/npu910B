# llm-perfkit

Benchmark toolkit for OpenAI-compatible LLM inference servers.

Point it at anything serving `/v1/chat/completions` — vLLM, SGLang, TGI,
llama.cpp server, or a hosted API — and get TTFT, TPOT, ITL, throughput and
**Goodput** under controlled load, plus Markdown / CSV / JSON reports you can
hand to someone else. Switching targets is a config change, nothing more.

Validated against both NVIDIA GPU and Ascend NPU deployments.

**English** · [中文](README.zh-CN.md)

## Why another benchmark tool

Most load generators report a single average latency and a request-per-second
number. For streaming LLM serving that hides the decisions you actually need to
make:

- **Averages hide the tail.** A p50 TTFT of 400 ms with a p99 of 12 s is a bad
  deployment that looks fine on paper. Everything here is reported as
  p50 / p90 / p95 / p99, never as a lone mean.
- **Raw throughput is not usable throughput.** Pushing concurrency up keeps
  total tokens/s climbing long after per-request latency has become unusable.
  Goodput — throughput counting *only* requests that met their SLO — is what
  tells you where to run in production.
- **Two throughput numbers move in opposite directions.** Server-wide output
  tokens/s rises with concurrency while per-request generation tokens/s falls.
  Quoting one when you meant the other is the most common way these numbers get
  misread, so both are always reported side by side.

See [docs/methodology.md](docs/methodology.md) for exact metric definitions and
the measurement caveats that come with them.

## Install

Requires Python 3.10+. The only hard dependency is `httpx`.

```bash
pip install llm-perfkit
```

Optional extras:

```bash
pip install "llm-perfkit[charts]"   # matplotlib, for plotting curves
pip install -e ".[dev]"             # pytest + ruff
```

From source:

```bash
git clone https://github.com/2309724680-cloud/npu910B.git
cd npu910B
pip install -e ".[dev]"
```

## Quick start

**1. Write a config.** Copy [examples/quickstart.json](examples/quickstart.json)
and fill in your endpoint:

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

`api_key` in `${VAR}` form is read from the environment, so no key is ever
written to disk. Leave it empty for an unauthenticated endpoint.

**2. Check the server is reachable** and see what it exposes:

```bash
perfkit probe -c conf.json
```

This reports the served model list, `max_model_len`, and whether `/metrics` and
`/tokenize` are available. Both are optional but change what can be measured:
without `/tokenize` input lengths are character-estimated, and without
`/metrics` there is no server-side cross-check of queue depth or KV usage.

**3. Smoke test** — three requests, confirms the whole path works before you
commit to a long run:

```bash
perfkit smoke -c conf.json
```

**4. Sweep concurrency:**

```bash
perfkit sweep -c conf.json --parallelism 1,4,8,16,32 \
    --input-tokens 1024 --output-tokens 128 --multiplier 10
```

Total requests per rung = `parallelism × multiplier`. The multiplier is fixed
rather than the total so every rung gets a sample count proportional to its
concurrency — with a fixed total, low rungs run many rounds and high rungs run
one, and their percentiles aren't comparable.

Any of the three dimensions accepts a list, so this is a 4 × 2 × 2 = 16-scenario
matrix:

```bash
perfkit sweep -c conf.json --parallelism 1,4,8,16 \
    --input-tokens 1024,4096 --output-tokens 128,512
```

**5. Or run scenarios defined in the config** — same matrix expansion applies to
list values there ([examples/concurrency-sweep.json](examples/concurrency-sweep.json)):

```bash
perfkit run -c conf.json
```

### Output

Each run writes to `out_dir`, prefixed with a run id:

| File | Contents |
| --- | --- |
| `<run>_requests.jsonl` | one line per request, written as it completes |
| `<run>_report.md` | full report: per-scenario tables, saturation analysis, caveats |
| `<run>_summary.csv` | one row per scenario, for spreadsheets |
| `<run>_curves.csv` | tidy data for the four standard curves |
| `<run>_summary.json` | the same aggregates, machine-readable |

The JSONL is written incrementally and flushed per record, so an interrupted run
still leaves usable data — you can re-aggregate from it without re-running.

With the `charts` extra installed, plot the curves:

```bash
python scripts/plot_curves.py results/<run>_curves.csv -o charts
```

### As a library

```python
import asyncio
from perfkit import ScenarioRunner, TokenCounter, load_config, aggregate

cfg = load_config("conf.json")
counter = TokenCounter(cfg.target)
runner = ScenarioRunner(cfg.target, counter, None, None, seed=cfg.seed)

step = asyncio.run(runner.run(cfg.scenarios[0], cfg.slo))
print(step.stats.ttft_ms["p95"], step.stats.goodput_rps)
```

## What it measures

| Metric | Meaning |
| --- | --- |
| **TTFT** | time to first token — prefill + queueing, what the user waits for |
| **TPOT** | time per output token, averaged over the decode phase |
| **ITL** | inter-token latency, the raw per-chunk gaps behind TPOT |
| **Output TPS** | server-wide output tokens/s on a wall-clock basis |
| **Generation TPS** | per-request decode speed, as one user experiences it |
| **Goodput** | requests/s that succeeded *and* met TTFT and TPOT SLOs |
| **Stall rate** | fraction of ITL gaps exceeding the stall threshold |

Load can be **closed-loop** (fixed number of requests in flight) or
**open-loop** (Poisson arrivals at a target rate). Prefix caching is controlled
explicitly per scenario — `unique`, `shared`, or `mixed` — since a shared prefix
can change TTFT by an order of magnitude and makes results incomparable if left
implicit.

When the target is vLLM, server-side `/metrics` is scraped before, during and
after each scenario so client-observed latency can be cross-checked against
actual queue depth and KV cache utilisation. This is what lets a report
distinguish "the model is slow" from "requests are waiting in line".

## Reading the results

Two traps worth knowing before you quote a number:

- **Use the median for per-request generation TPS.** A stream that flushes all
  its tokens in one chunk yields an absurd per-request rate and drags the mean
  far off. The reports carry both; the median is the honest one.
- **`/metrics` gauges are cumulative extrema.** Peak queue depth from a
  `/metrics` diff reflects the highest value since server start, not this
  scenario's peak, unless you recompute per time window.

Both are called out in the generated report as well, so a reader who didn't run
the benchmark still sees them.

## Development

```bash
pip install -e ".[dev]"
ruff check .
pytest
```

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
