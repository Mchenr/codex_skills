# CPU Target Workflow

## Inputs To Resolve

- `MODEL_NAME`: usually `wd_dcn`, `din_mmoe`, or `bst_mmoe`.
- `CONTAINER`: target inference container, for example `benchmark_infer_workspace_32t`.
- `CPU_QUOTA_CORES`: read from cgroup quota. For 32T this should be `32`.
- `TARGET_CPU_PCT`: target normalized quota percentage, for example `60`.
- `TARGET_CORES`: `CPU_QUOTA_CORES * TARGET_CPU_PCT / 100`.
- `MODEL`: `/workspace/<MODEL_NAME>/1` unless the user gives another path.
- `DATA`: `/workspace/benchmark_dataset/<MODEL_NAME>.tsv` unless the user gives another path.
- `OUT_DIR`: unique path under `/workspace/benchmark_results`.
- For KDNN/DNN capacity tests, prefer low-pressure prewarm before measured load:
  `PREWARM_QPS=500`, `PREWARM_DURATION=30`, and measured-run `WARMUP=0`.

## CPU Quota Check

Run inside the container:

```bash
cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null
cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null
cat /sys/fs/cgroup/cpu.max 2>/dev/null || true
```

For cgroup v1, quota cores are `cpu.cfs_quota_us / cpu.cfs_period_us`.

## Coarse Sweep Template

Use this shape for a coarse sweep. Adjust `QPS_LIST`, `PORT`, and thread settings to match the request.

```bash
docker exec <CONTAINER> bash -lc '
cd /workspace/tensorflow &&
MODEL_NAME=<MODEL_NAME> \
MODEL=/workspace/<MODEL_NAME>/1 \
DATA=/workspace/benchmark_dataset/<MODEL_NAME>.tsv \
OUT_DIR=/workspace/benchmark_results/<MODEL_NAME>_<SPEC>_cpu<TARGET>_coarse \
PORT=<PORT> \
THREAD_NUM=<SERVER_THREADS> \
CLIENT_THREAD_NUM=<CLIENT_THREADS> \
TF_NUM_INTRAOP_THREADS=<INTRA> \
TF_NUM_INTEROP_THREADS=<INTER> \
CPU_QUOTA_CORES=<CPU_QUOTA_CORES> \
PREWARM_QPS=<PREWARM_QPS> \
PREWARM_DURATION=<PREWARM_DURATION> \
WARMUP=<MEASURED_WARMUP> \
DURATION=30 \
MAX_INFLIGHT=300 \
QPS_LIST="<QPS_LIST>" \
REQUEST_MIN_BATCH_SIZE=1 \
REQUEST_MAX_BATCH_SIZE=100 \
TIMEOUT_MS=60000 \
MACHINE_LABEL=<SPEC>_cpu<TARGET>_coarse \
BIN_SERVER=/usr/local/bin/predictor_server \
BIN_CLIENT=/usr/local/bin/brpc_client \
./tools/run_serving_benchmark.sh
'
```

Initial 32T `wd_dcn` QPS guesses can start with:

```text
500 1000 1500 2000 2500 3000 3500 4000
```

If CPU is still far below target and failure/dropped remain zero, extend upward. If dropped appears before target CPU, increase `MAX_INFLIGHT` and/or client pressure.

For KDNN/DNN runs, use low-pressure prewarm unless the user explicitly asks for cold-start behavior:

```text
PREWARM_QPS=500
PREWARM_DURATION=30
WARMUP=0
```

This makes the prewarm use 500 QPS and keeps the measured run clean. Regular `WARMUP=10` is not equivalent because the client warms up at the target QPS, which can overload a cold KDNN server.

## Fine Sweep

Choose QPS values around the two coarse points that bracket the target. Example:

```text
2600 2700 2800 2900 3000
```

Keep all other parameters identical to the coarse sweep.

## Anomaly Handling

Use abnormal-data handling before selecting the target point:

- Abnormal signs: nonzero `Failure` or `Dropped`, CPU avg far from nearby points, CPU max much higher than CPU avg, latency spikes that do not match the QPS trend, or suspected host load interference.
- For KDNN, abnormal first-point behavior can come from starting the measured load too hard. In a 32T `wd_dcn` test, fresh `2200 QPS` without low-pressure prewarm had `CPU avg=69.60% quota` and `CPU max=99.61% quota`; with `PREWARM_QPS=500 PREWARM_DURATION=30 WARMUP=0`, the measured `2200 QPS` run had `CPU avg=62.40% quota`, `CPU max=63.72% quota`, `failure=0`, and `dropped=0`.
- When a point is abnormal, re-run the exact same mode, QPS, thread, batch, and inflight settings once in a new `OUT_DIR` with a `_rerun` suffix.
- If the abnormal point is a KDNN first measured point and no low-pressure prewarm was used, first re-run it with low-pressure prewarm. Treat that as the proper validation run.
- If the re-run is normal, use the re-run result and mention the discarded abnormal result.
- If the re-run is also abnormal, stop the search and report both runs. Do not continue to higher QPS or interpolate through the abnormal region.

## Stability Run

Run only the best QPS point with longer timing:

```bash
WARMUP=20
DURATION=120
QPS_LIST="<BEST_QPS>"
OUT_DIR=/workspace/benchmark_results/<MODEL_NAME>_<SPEC>_cpu<TARGET>_stable
```

For KDNN stability runs, usually use:

```bash
PREWARM_QPS=500
PREWARM_DURATION=30
WARMUP=0
DURATION=120
```

Use the stable run as the final answer unless it has failures, drops, or CPU drift far from target.

## Reading Results

Read:

```bash
cat <host_out_dir>/metadata.txt
cat <host_out_dir>/results.md
```

The important columns are:

- `Server CPU avg (% quota)`: normalized CPU percentage against `CPU_QUOTA_CORES`.
- `Server CPU avg cores`: actual average core consumption.
- `P99 latency (us)`, `avg_latency_us`, `Failure`, and `Dropped`.

Final report format:

```markdown
| Mode | Target CPU % | QPS | P99 us | Avg us | CPU avg % quota | CPU avg cores | Failure | Dropped |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 60 | ... | ... | ... | ... | ... | ... | ... |
```

State if the target was bracketed, interpolated, or not reached.
Also state whether any abnormal point was re-tested and whether it was accepted or rejected.
