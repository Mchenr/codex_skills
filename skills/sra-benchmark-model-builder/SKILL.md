---
name: sra-benchmark-model-builder
description: Use when building or modifying SRA benchmark TensorFlow SavedModels for predictor-server hardware inference benchmarking, especially when deriving train/export scripts and inference TSV generators from pbtxt/reference graphs such as wd_dcn or wide_and_deep.
---

# SRA Benchmark Model Builder

## Purpose

Build benchmark-only TensorFlow 1.x SavedModels and TSV inference data for predictor-server serving tests under `/home/c00913906/sra_benchmark`. The priority is to match the reference graph's inference signature, op mix, tensor shapes, and serving behavior for hardware performance evaluation. Model accuracy is secondary unless the user explicitly asks otherwise.

## Required Context

- Read `/home/c00913906/AGENT.md` before changing model code or running training.
- If the target follows the completed `wd_dcn` work, read `references/wd_dcn-pattern.md`.
- Keep repository work inside `/home/c00913906/sra_benchmark` unless the user explicitly expands scope.
- Use `/home/c00913906/tmp` on the host and `/workspace/tmp` inside containers. Avoid `/tmp` because the root filesystem can be full.
- Use the `benchmark-train-dev` container for model training and SavedModel validation unless the user names a different runtime.
- After SavedModel validation, if the `benchmark-infer:v1.0.0` image is present, run one predictor-server validation with that image's `brpc_server` environment and collect one TensorFlow timeline sample. Follow `tensorflow/BUILD_GUIDE.md` for the current serving binary names and client flags.
- Use `/home/c00913906/models` as the default downloaded model artifact root and `/home/c00913906/dataset` as the default downloaded inference dataset root. In predictor containers these resolve through the host mount as `/data/models` and `/data/dataset`.
- Run AtomGit download/upload on the host or in the predictor container only; do not run AtomGit in `benchmark-train-dev` because Python 3.7 and AtomGit are incompatible there.
- Never write AtomGit credentials into files. Use the local `atomgit` login state or an environment variable such as `ATOMGIT_TOKEN`; if a token is pasted into chat, tell the user to rotate it.

## Workflow

1. Locate or create `modelzoo/<model_name>`.
2. Inspect the reference pbtxt or graph for input tensors, output tensors, feature families, key op counts, layer widths, sparse/dense boundaries, and graph-only serving requirements. Before writing data generators, produce a concrete input feature inventory from the reference graph.
3. Create or maintain a benchmark-specific TF1 training/export script:
   - If no benchmark-specific training script exists for the target model/version, add a new script under `modelzoo/<model_name>` instead of modifying the original upstream project training script.
   - Only adapt an existing script when it was already created for this predictor-server benchmark workflow.
   - Keep upstream/demo model scripts intact unless the user explicitly asks to change them.
   - Include synthetic-data support, configurable feature counts, deterministic seeds, and enough training steps to materialize variables.
4. Export a clean inference-only SavedModel by training to a checkpoint, resetting the graph, rebuilding the inference graph only, restoring variables, and saving `saved_model/1`.
5. Implement or adapt `tools/gen_infer_data.py` to generate TSV records with the schema `id<TAB>model_input`, using the feature families discovered from the reference graph rather than assuming the `wd_dcn` four-field format.
6. Train inside `benchmark-train-dev`, writing outputs under `/workspace/modelzoo/<model_name>/result/<version>`.
7. Validate the result: load the SavedModel, smoke-test inference, inspect the serving signature, and count important ops from the SavedModel graph.
8. If `benchmark-infer:v1.0.0` exists locally, validate through that image's `brpc_server` runtime: run `predictor_server` against the SavedModel, run `brpc_clinet` against the generated TSV, and collect one timeline sample.
9. If an artifact already exists remotely, download it with `scripts/atomgit_artifacts.py` by `--model-name` before rebuilding.
10. Update `/home/c00913906/AGENT.md` with the model path, TSV path, exact training command, concrete input feature inventory, SavedModel validation, predictor-server validation, and timeline path.
11. Upload completed model and dataset artifacts to AtomGit with `scripts/atomgit_artifacts.py` when the user asks to publish them.


## AtomGit Artifacts

Remote artifact repositories:

- Models: `codesheepchen/benchmark`
- Inference datasets: `codesheepchen/benchmark_dataset`

Default local roots:

- Host model root: `/home/c00913906/models`
- Host dataset root: `/home/c00913906/dataset`
- Predictor-container model root: `/data/models`
- Predictor-container dataset root: `/data/dataset`

Use the helper script on the host or in the predictor container when the user asks to fetch or publish artifacts by model name:

```bash
python /home/c00913906/.codex/skills/sra-benchmark-model-builder/scripts/atomgit_artifacts.py \
  download-model --model-name <model_name>

python /home/c00913906/.codex/skills/sra-benchmark-model-builder/scripts/atomgit_artifacts.py \
  download-dataset --model-name <model_name>

python /home/c00913906/.codex/skills/sra-benchmark-model-builder/scripts/atomgit_artifacts.py \
  upload-model --model-name <model_name> --source <saved-model-or-version-dir>

python /home/c00913906/.codex/skills/sra-benchmark-model-builder/scripts/atomgit_artifacts.py \
  upload-dataset --model-name <model_name> --source <dataset-dir-or-file>
```

Equivalent raw AtomGit commands are:

```bash
atomgit download codesheepchen/benchmark -d ./models
atomgit download codesheepchen/benchmark_dataset -d ./dataset
atomgit upload ./your-model-dir --repo-id codesheepchen/benchmark
atomgit upload ./your-dataset-dir --repo-id codesheepchen/benchmark_dataset
```

The helper downloads the repository into the root and reports the expected path for the requested `model_name`. If the repository layout differs, pass `--source` for uploads and record the actual resolved path in `AGENT.md`.

## Graph Rules

- Prefer a single string input named `model_input` for predictor-server compatibility.
- Derive the model-input format from the reference pbtxt or graph. Do not assume every model has the `wd_dcn` four feature families.
- Use `wd_dcn`'s `g=<float>;t=<ids>;w=<ids>;p=<ids>` format only when the workload actually has global, deep/trival, wide, and plain feature families.
- For each model, define and record the exact input feature inventory: feature-family key in `model_input`, semantic role, scalar/id/list type, count or shape, bucket/hash range, embedding dimension if applicable, delimiter/encoding, and which branch or subgraph consumes it.
- Keep exported serving graphs free of training, optimizer, and gradient nodes.
- Preserve expensive op structure from the reference graph rather than maximizing TensorFlow estimator convenience.
- For `wd_dcn`-style loads, the clean serving graph should have exactly two `SparseTensorDenseMatMul` nodes: one at the first deep dense layer and one at the first wide MLP dense layer. Later MLP layers should use `MatMul` unless the reference graph requires otherwise.
- Avoid `BiasAdd` when matching the current predictor-server expectation; use explicit `AddV2` style bias addition if needed.
- Parse SavedModel protobufs as SavedModels, not as raw GraphDefs.

## Validation Commands

Use container commands with an explicit temp directory, for example:

```bash
docker exec benchmark-train-dev bash -lc 'cd /workspace/modelzoo/<model_name> && TMPDIR=/workspace/tmp python train.py --output_dir=./result/<version> ...'
```

Smoke-test a SavedModel from outside its source tree:

```bash
docker exec benchmark-train-dev bash -lc 'cd /workspace && TMPDIR=/workspace/tmp python - <<"PY"
import tensorflow as tf
path = "/workspace/modelzoo/<model_name>/result/<version>/saved_model/1"
with tf.Session(graph=tf.Graph()) as sess:
    meta = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], path)
    sig = meta.signature_def[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    print(sig)
PY'
```

When validation finds mismatched op counts, adjust the graph construction first; do not hide mismatches in the inference data generator.

## Required Output Notes

Every completed model build must list the concrete input feature inventory in the final response and in `/home/c00913906/AGENT.md`. Include one row or bullet per feature family, for example:

- `g`: global scalar features, float, shape `[batch, 1]`, consumed by global branch.
- `t`: deep/trival id list, integer ids, count `128`, bucket `600000`, embedding dim `1668`, consumed by deep branch.

Use the model's actual reference-derived feature families. The `g/t/w/p` example is only for `wd_dcn`-style models.

## Predictor-Server Validation

After SavedModel smoke validation, check whether the benchmark inference image is available:

```bash
docker image inspect benchmark-infer:v1.0.0 >/dev/null 2>&1
```

If the image is available, run a predictor-server validation using that image's environment. Prefer an existing running container from `benchmark-infer:v1.0.0`; otherwise start a temporary container from that image following `tensorflow/BUILD_GUIDE.md`:

```bash
docker run -d \
    --name brpc_server \
    --cpuset-cpus=0-79 \
    --cpuset-mems=0 \
    --network=host \
    -v /home/workspase:/data \
    benchmark-infer:v1.0.0 \
    tail -f /dev/null
```

Adjust the host mount only if the active workspace differs from `/home/workspase`. Do not use a different image merely because a container is named `brpc_server`; verify the image first.

Build the current server/client targets inside the inference container if needed:

```bash
cd /data/tensorflow
bazel --output_user_root=./output build --enable_bzlmod=true --experimental_enable_bzlmod --experimental_repo_remote_exec --cxxopt=-std=c++17 --host_cxxopt=-std=c++17 --copt=-march=armv8.5-a --host_copt=-I/usr/local/include --linkopt=-L/usr/local/lib64 --host_linkopt=-L/usr/local/lib64 --linkopt=-Wl,-rpath,/usr/local/lib64 --host_linkopt=-Wl,-rpath,/usr/local/lib64 --action_env=LD_LIBRARY_PATH=/usr/local/lib64:${LD_LIBRARY_PATH:-} --action_env=TF_SYSTEM_LIBS="boringssl,snappy" --distdir=/data/download --check_direct_dependencies=off //:predictor_server
bazel --output_user_root=./output build --enable_bzlmod=true --experimental_enable_bzlmod --experimental_repo_remote_exec --cxxopt=-std=c++17 --host_cxxopt=-std=c++17 --copt=-march=armv8.5-a --host_copt=-I/usr/local/include --linkopt=-L/usr/local/lib64 --host_linkopt=-L/usr/local/lib64 --linkopt=-Wl,-rpath,/usr/local/lib64 --host_linkopt=-Wl,-rpath,/usr/local/lib64 --action_env=LD_LIBRARY_PATH=/usr/local/lib64:${LD_LIBRARY_PATH:-} --action_env=TF_SYSTEM_LIBS="boringssl,snappy" --distdir=/data/download --check_direct_dependencies=off //:brpc_clinet
```

Inside the inference container, use paths under `/data/sra_benchmark`:

```bash
cd /data/tensorflow
TF_NUM_INTEROP_THREADS=16 TF_NUM_INTRAOP_THREADS=16 numactl -C 0-15 \
  ./bazel-bin/predictor_server \
  --enable_kdnn=false \
  --model_path=/data/sra_benchmark/modelzoo/<model_name>/result/<version>/saved_model/1 \
  --thread_num=16
```

From the same inference container, run the client against the generated TSV:

```bash
cd /data/tensorflow
./bazel-bin/brpc_clinet \
  --server=127.0.0.1:8000 \
  --input_data_path=/data/sra_benchmark/modelzoo/<model_name>/result/<version>_infer.tsv \
  --thread_num=16 \
  --test_duration_s=10 \
  --warmup_duration_s=5 \
  --max_qps=1000 \
  --max_inflight=100 \
  --request_min_batch_size=1 \
  --request_max_batch_size=100
```

`brpc_clinet --max_qps` is global RPC QPS, not per-thread QPS; `--max_inflight` limits concurrent in-flight RPCs and overflow is counted as dropped/failure. `--request_min_batch_size` and `--request_max_batch_size` control how many samples are wrapped into each RPC.

Then collect one TensorFlow timeline sample by restarting the server with timeline enabled:

```bash
mkdir -p /data/timeline/<model_name>/<version>
cd /data/tensorflow
TF_NUM_INTEROP_THREADS=16 TF_NUM_INTRAOP_THREADS=16 numactl -C 0-15 \
  ./bazel-bin/predictor_server \
  --enable_kdnn=false \
  --model_path=/data/sra_benchmark/modelzoo/<model_name>/result/<version>/saved_model/1 \
  --thread_num=16 \
  --enable_tf_timeline=true \
  --tf_timeline_every_n=100 \
  --tf_timeline_max_dumps=1 \
  --tf_timeline_dump_warmup=false \
  --tf_timeline_dir=/data/timeline/<model_name>/<version>
```

Run the client once more while the timeline-enabled server is running. Record in `AGENT.md`:

- The inference container name and image.
- The exact server and client commands.
- Client success or failure and any observed QPS/latency output.
- Timeline output directory and generated `*.runmeta.pb` files.

Convert a collected RunMetadata protobuf to TensorFlow Chrome trace JSON with the script documented in `tensorflow/BUILD_GUIDE.md`:

```bash
python /workspace/tools/runmetadata_to_timeline.py /workspace/runmeta_convert/<file>.runmeta.pb
```
