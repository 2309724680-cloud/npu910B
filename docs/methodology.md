# Benchmark methodology

What each metric means, how it is computed, and what it does not tell you. If
you only read one section, read [SLO and Goodput](#slo-and-goodput).

## Contents

- [Timing model](#timing-model)
- [Metric definitions](#metric-definitions)
- [Percentiles, not averages](#percentiles-not-averages)
- [SLO and Goodput](#slo-and-goodput)
- [Load models](#load-models)
- [Saturation](#saturation)
- [Prompt construction](#prompt-construction)
- [Prefix caching](#prefix-caching)
- [Server-side metrics](#server-side-metrics)
- [Reliability accounting](#reliability-accounting)
- [Known limitations](#known-limitations)

## Timing model

Everything is measured client-side with a monotonic clock, from the process
issuing the requests. Streaming is mandatory: TTFT, ITL and TPOT are only
definable if you can observe the arrival time of each chunk. A scenario with
`stream: false` is rejected at validation rather than silently downgraded — if
you need only end-to-end latency and throughput, that is a different
measurement and should be labelled as one.

Four timestamps per request:

```
t_send ──────────► t_first_token ──────► … chunks … ──────► t_last_token
        TTFT                    ITL gaps
        └──────────────────── E2E ─────────────────────────────────┘
```

`t_send` is taken immediately before the HTTP write. TTFT therefore includes
connection reuse, request serialisation, network transit, server queueing and
prefill — everything the user actually waits through, not just model compute.
That is deliberate: separating them requires server-side instrumentation, which
[server-side metrics](#server-side-metrics) covers where available.

Client-side measurement has a floor set by the client itself. Under high
concurrency the event loop can add scheduling delay to the observed chunk
arrival time, which shows up as ITL noise at the sub-millisecond scale. It does
not affect conclusions at the tens-of-milliseconds scale where TPOT lives, but
it does mean a single ITL sample is not meaningful — only the distribution is.

## Metric definitions

Let a request produce `N` output tokens with chunk arrival times
`c_1 … c_N` (a chunk may carry more than one token; see the caveat below).

| Metric | Definition |
| --- | --- |
| **TTFT** | `c_1 - t_send`, in ms |
| **E2E** | `c_N - t_send`, in ms |
| **ITL** | the set of gaps `c_i - c_{i-1}` for `i > 1`, in ms |
| **TPOT** | `(c_N - c_1) / (N - 1)`, in ms — mean decode cost per token |
| **Generation TPS** | `(N - 1) / (c_N - c_1)` — per-request decode rate |
| **Output TPS** | `Σ N / wall_clock` over the scenario — server-wide rate |
| **Stall rate** | fraction of ITL gaps exceeding `stall_itl_ms` |

Three things to note.

**TPOT excludes prefill by construction.** It measures from the first token, not
from send. TPOT and TTFT are separate concerns — prefill is compute-bound and
scales with input length, decode is bandwidth-bound and largely is not — and a
single latency number that blends them cannot be reasoned about.

**Requests with `N < 2` have no TPOT, ITL or generation TPS.** They are counted
for success and TTFT and excluded from the decode statistics, rather than
contributing a zero that would drag the distribution down.

**Output TPS and generation TPS move in opposite directions.** Raising
concurrency raises server-wide Output TPS (more requests share each batch) while
lowering per-request Generation TPS (each request gets a smaller share of
bandwidth). Both are real; they answer different questions — "how much can this
box serve" versus "how fast does it feel". Quoting one under the other's name is
the single most common way these numbers get misreported, so every report
carries both.

### The chunk-vs-token caveat

ITL is computed per *chunk*, and a server may emit several tokens per SSE frame.
When it does, ITL over-reports the gap and the token count per chunk is what
actually varies. TPOT is computed from total tokens over total decode time, so
it stays correct regardless of chunking; ITL and stall rate are the
chunking-sensitive ones. If a target batches tokens into chunks aggressively,
treat stall rate as an upper bound.

A related artifact: some servers flush an entire short response in one chunk. For
those records `E2E ≈ TTFT` and generation TPS becomes an enormous number that is
arithmetically true and practically meaningless. A handful of such records is
enough to move an arithmetic mean by an order of magnitude, which is why
per-request TPS is reported and should be read as a **median**.

## Percentiles, not averages

Every latency metric is reported as p50 / p90 / p95 / p99 plus min, max, mean,
std and count. Percentiles use linear interpolation between order statistics
(the standard "linear" / type-7 method), so `p90` of `1…10` is `9.1`.

The mean is included for completeness and should almost never be the number you
quote. LLM serving latency distributions are right-skewed by queueing: a
deployment with an acceptable p50 and an unacceptable p99 is a deployment that
fails for a minority of every user's requests, which is how it will be
experienced.

Percentile confidence depends on sample count. p99 from 30 requests is noise. In
concurrency sweeps the request count per rung is `parallelism × multiplier`
precisely so that sample size scales with concurrency; if you override it with a
fixed per-rung total, low rungs end up with fewer samples relative to their
duration and their tail percentiles carry less confidence than high rungs'. The
generated report notes when this has happened.

## SLO and Goodput

Throughput without a latency bound is not a capacity number. A server will keep
accepting requests and keep producing tokens long past the point where anyone
would accept the latency, so "maximum tokens/s" tends to land at a concurrency
you would never deploy at.

**Goodput** is throughput counted over successful requests only:

```
goodput_rps = |{ r : r.ok ∧ r.ttft ≤ SLO_ttft ∧ r.tpot ≤ SLO_tpot }| / wall_clock
```

with an optional E2E bound. Compliance rate — the same numerator over all
requests — is reported alongside it, because a Goodput of 0.6 req/s means
something very different at 95% compliance than at 20%.

Goodput has a maximum, and it is usually well below the concurrency that
maximises raw throughput. That maximum is the interesting number: past it,
additional concurrency converts into queueing delay and SLO violations rather
than useful work.

### Setting the SLO

The defaults (`ttft_ms: 2000`, `tpot_ms: 100`, `stall_itl_ms: 500`) are a
fallback so the tool produces a Goodput number at all. They are not a
recommendation. Real thresholds come from the product:

- **TTFT** is what a user waits before seeing anything. Interactive chat wants
  sub-second; batch summarisation can tolerate tens of seconds.
- **TPOT** sets the streaming rate. 100 ms/token is ~10 tokens/s, roughly slow
  reading speed. 50 ms/token feels fluent.
- **Stall threshold** catches perceptible pauses mid-stream that a TPOT average
  hides entirely — one 2-second gap in an otherwise fast stream is a visible
  glitch that barely moves the mean.

**Set SLOs per length bucket, not globally.** TTFT scales with input length
because prefill does. A 2000 ms bound is generous at 1K input and physically
impossible at 128K on the same hardware; applying it to both makes the long
bucket read as total failure and tells you nothing. Each scenario can override
the run-level SLO with a `slo` block, and every report states which thresholds
were applied to which scenario.

## Load models

Two load models, answering different questions. Choose deliberately — they can
give opposite answers about the same server.

**Closed loop** (`mode: "closed"`) holds `parallelism` requests in flight; a new
one is issued as each completes. This is the right model for capacity planning
and for comparing configurations, because the load is self-limiting: if the
server slows down, the client issues requests more slowly, so the queue cannot
grow without bound and latency stays interpretable. It is what a fixed-size
worker pool or a connection-limited gateway does.

The limitation is that it cannot show you overload. A closed-loop client at
parallelism 32 never puts more than 32 requests on the server, no matter how
badly it is coping.

**Open loop** (`mode: "open"`) issues requests at a target rate with Poisson
inter-arrival times, independent of whether previous requests have finished. This
is what real traffic looks like, and it is the only way to observe the
instability that matters: if the arrival rate exceeds service capacity, the queue
grows without bound and latency diverges rather than settling. Poisson (rather
than fixed-interval) arrivals matter because the bursts in a Poisson stream are
what actually trigger queueing — evenly spaced arrivals at the same mean rate
understate delay.

Practical guidance: sweep closed-loop to find capacity and the shape of the
latency curve, then confirm with open-loop at the arrival rate implied by your
chosen operating point. If open-loop latency diverges at a rate the closed-loop
sweep suggested was fine, the closed-loop result was hiding a queue.

## Saturation

`find_saturation` looks for three knees across a concurrency ladder and reports
each separately, because they generally do not coincide:

- **Throughput knee** — the first rung where total output TPS gains less than
  `TPS_KNEE_GAIN` (5% by default) over the previous rung. Past this, added
  concurrency buys almost no additional throughput.
- **Goodput knee** — the first rung where Goodput *decreases*. This is the
  operating ceiling: beyond it you are trading successful requests for
  unsuccessful ones.
- **SLO knee** — the first rung where p95 TTFT or p95 TPOT crosses its SLO.

Comparison is only valid within one `(input_tokens, output_tokens)` group. A
ladder that mixes length combinations would compare rungs whose work per request
differs, so groups other than the one with the most rungs are excluded and named
in the output rather than silently folded in.

Three caveats on interpreting the result:

**A ceiling may be the client's or the server's config, not the hardware's.** If
the top rung of your ladder equals the server's `max_num_seqs`, the flattening
you see is that limit, not a saturated accelerator. Check it before concluding
anything about the hardware.

**The knee is often a shoulder.** Real curves flatten over a range of rungs
rather than at a point. The reported knee is the first rung meeting the
criterion; read the curve, not just the number.

**Attribute the bottleneck by elimination.** Flat throughput with low memory
utilisation, low compute utilisation and a TTFT step that lands on a round token
boundary points at a batching or scheduling limit (e.g. a max-batched-tokens
cap), not at the hardware. Collecting device-level utilisation alongside the
benchmark is what makes that distinction possible.

## Prompt construction

`input_tokens` is a target, approached iteratively: text is generated, tokenised,
and grown or trimmed until the count is close to target. Where the server exposes
`/tokenize`, that endpoint is authoritative, so lengths match the server's own
accounting exactly. Without it, lengths are estimated from character count and
will differ from the server's tokenisation — the report records which source was
used, and cross-target comparisons made on estimates are not sound.

Two consequences:

- **Report measured lengths, not target lengths.** Convergence stops within a
  small tolerance of the target, so actual input can land a couple of percent
  below the requested value. At 128K that is a visible absolute gap. Reports
  carry the measured mean.
- **Content is deliberately low-information filler.** Prompts are synthetic and
  seeded (`seed` in the config) so runs are reproducible. This measures serving
  mechanics, not model quality, and says nothing about output usefulness.

Output length is controlled by `max_tokens` plus `ignore_eos: true`, which forces
the model to generate exactly the requested count. Without `ignore_eos`, output
length is whatever the model decides, varies per request, and makes per-token
metrics much noisier — set it false only when you specifically want realistic
length distributions, and expect wider spread.

## Prefix caching

Prefix cache state changes TTFT by an order of magnitude, so it is set explicitly
rather than left to chance:

- `unique` — every prompt gets a distinct prefix. No cache reuse. This is the
  conservative, worst-case-prefill number and the right default for capacity
  planning.
- `shared` — all prompts share a common prefix, so all but the first hit cache.
  Best case. Represents a system-prompt-heavy workload.
- `mixed` — `shared_prefix_ratio` of the prompt is shared, the rest unique.
  Closest to production for an assistant with a fixed system prompt.

A benchmark that does not state its prefix mode is not comparable with one that
does. The mode appears in every report and in `curves.csv`.

Note that cache state persists on the server between scenarios. A `unique` run
immediately after a `shared` run may still benefit from residual cache; warmup
requests (`warmup_requests`, excluded from statistics) exist partly to bring the
server to a consistent state, but for strict isolation restart or flush the
server between runs.

## Server-side metrics

When the target exposes Prometheus `/metrics` (vLLM does), it is scraped before,
during and after each scenario. This is what lets a report separate "the model is
slow" from "requests are queued": high TTFT with a deep queue is a scheduling
problem, high TTFT with an empty queue is a compute problem.

Collected: request queue depth, running/waiting counts, KV cache utilisation,
and cumulative token counters. The cumulative counters are diffed across the
scenario, so server-side token totals can be cross-checked against
client-observed ones — a mismatch means one of the two is being counted wrong,
and it is worth knowing which.

**Gauge peaks are cumulative extrema.** A peak read from the start/end pair
reflects the highest value since server start, not this scenario's peak. When
several scenarios run in sequence, later ones inherit earlier peaks. For
per-scenario peaks, set `metrics_interval_s` to sample throughout and compute
over the scenario's own window. Reports flag the peak columns accordingly.

## Reliability accounting

Failures are counted by type, not as a single total. Connection errors, HTTP
status errors, timeouts, malformed SSE frames and truncated streams have
different causes and different fixes, and a single failure count hides which one
you have.

Some failures are deployment-correctness problems rather than performance
results — a model name mismatch, a context-length rejection, an auth failure.
Those invalidate the run rather than degrade it, and reading them as a latency
result is a mistake. `perfkit probe` and `perfkit smoke` exist to catch them
before a long run.

Every request is written to JSONL as it completes, flushed per record. An
interrupted run leaves usable data that can be re-aggregated without re-running.

## Known limitations

Stated plainly, because a benchmark that hides these is worse than no benchmark:

- **Client-side only.** TTFT includes network transit and cannot be decomposed
  into queueing versus prefill without server-side instrumentation. Run the
  client close to the server, and note the topology in the report.
- **Single client process.** At very high concurrency the client's own event loop
  becomes a factor. If client CPU is saturated, measured latency includes it.
  Check client load before trusting a high-concurrency result.
- **Synthetic prompts.** Measures serving mechanics, not model quality or
  real-workload token distributions.
- **Cross-run comparisons need matched conditions.** Prefix mode, `ignore_eos`,
  measured (not target) lengths, SLO thresholds, and server config must match.
  The report records all of these so a reader can check.
- **Device utilisation is not collected automatically.** Accelerator
  utilisation, memory and power come from a vendor tool (`nvidia-smi`,
  `npu-smi`) run alongside the benchmark, and must be correlated by timestamp.
  Verify host clocks agree first — a clock offset between the benchmark host and
  the server silently misaligns the two series, and the misalignment is easy to
  mistake for a real effect.


