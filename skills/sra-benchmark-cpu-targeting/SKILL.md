---
name: sra-benchmark-cpu-targeting
description: Find the SRA benchmark model QPS and latency at a target normalized container CPU load, such as "32T 60% CPU", "CPU load target", "capacity point", "QPS at target CPU", or "measure model throughput under a CPU budget". Use for TensorFlow predictor_server/brpc_client benchmark sweeps on wd_dcn, din_mmoe, bst_mmoe, or similar SRA SavedModel workloads when Codex must choose coarse and fine QPS ranges, validate CPU quota normalization, run/interpret benchmark results, and report the target CPU capacity point.
---

# SRA Benchmark CPU Targeting

## Objective

Find the QPS and latency corresponding to a target container CPU load by sweeping client QPS, using CPU normalized to the container quota rather than raw process `%CPU`.

Use the current workspace conventions unless the user says otherwise:

- Host workspace: `/home/c00913906`; container path: `/workspace`.
- Standard benchmark script: `/workspace/tensorflow/tools/run_serving_benchmark.sh`.
- Preferred binaries for standard runs: `/usr/local/bin/predictor_server` and `/usr/local/bin/brpc_client`.
- Preferred data root: `/workspace/benchmark_dataset`.
- Runtime model root: `/workspace/<model>/1`.

## Required Checks

Before running a target-CPU benchmark:

1. Confirm the target container is running and read its CPU quota.
2. Confirm model and TSV paths exist.
3. Confirm `run_serving_benchmark.sh` outputs CPU normalized by `cpu_quota_cores`.
4. Set `TF_NUM_INTRAOP_THREADS`, `TF_NUM_INTEROP_THREADS`, `THREAD_NUM`, `CLIENT_THREAD_NUM`, request batch range, and `MAX_INFLIGHT` explicitly.
5. Use a unique `OUT_DIR` containing model, container spec, target CPU, and sweep type.

For 32T, the 60% target is `19.2` cores. Results should use `Server CPU avg (% quota)` around `60.00`, not raw process `%CPU` around `1920%`.

## Workflow

1. **Coarse sweep**
   Run a broad `QPS_LIST` to bracket the target CPU. Prefer QPS values wide enough to cross the target. Increase `MAX_INFLIGHT` if the client drops requests before CPU reaches the target.

2. **Fine sweep**
   Run a narrower QPS list around the closest coarse points. Keep all non-QPS parameters identical.

3. **Stability run**
   Re-run the best QPS point with longer warmup and duration. Use this as the final capacity point.

4. **Anomaly handling**
   If a point is abnormal, re-run the same parameters once. If the re-run is still abnormal, stop the target search and report the anomaly instead of continuing with distorted data.

5. **Low-pressure prewarm**
   For KDNN/DNN runs, separate server prewarm from the measured load. Use low-pressure prewarm such as `PREWARM_QPS=500 PREWARM_DURATION=30 WARMUP=0` before the measured QPS so the first measured point is not polluted by cold KDNN/server startup behavior. Do not treat regular `WARMUP` as low-pressure prewarm, because it uses the same target QPS and can still overload a cold server.

6. **Report**
   Include the result directory, model, data path, container quota cores, intra/inter, server/client thread counts, request batch range, QPS, P99, avg latency, CPU avg/max `% quota`, CPU avg/max cores, failure, and dropped.

Read [references/cpu-target-workflow.md](references/cpu-target-workflow.md) when generating commands, selecting QPS ranges, or interpreting edge cases.

## Interpretation Rules

- If CPU never approaches the target and failures/drops appear, increase `MAX_INFLIGHT` or client pressure before concluding model capacity.
- If CPU never approaches the target but latency explodes, treat it as a non-CPU bottleneck and inspect server logs/timeline.
- Treat a result as abnormal when failure/dropped appears unexpectedly, CPU avg differs sharply from neighboring points or prior repeat runs, CPU max is much higher than avg, latency spikes out of line with QPS, or host load is suspected. Re-test the same point once before using or rejecting it.
- For KDNN, suspect cold-start pressure if the first measured point has CPU max near 100% quota or drops requests while later higher-QPS points look normal. Re-run with low-pressure prewarm before rejecting the point.
- If the repeated point is still abnormal, exit the benchmark search and report both runs, the suspected cause, and the last valid result. Do not silently skip forward to higher QPS.
- If DNN and baseline are both requested, find and report target points separately for each mode.
- Do not compare CPU values from old results unless their `results.md` explicitly says `% quota` or metadata includes `cpu_quota_cores`.
- Do not use `/workspace/dataset` unless the user explicitly asks; use `/workspace/benchmark_dataset`.
