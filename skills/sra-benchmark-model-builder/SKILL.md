---
name: sra-benchmark-model-builder
description: Use when building or modifying SRA benchmark TensorFlow SavedModels for predictor-server hardware inference benchmarking, especially when deriving train/export scripts and inference TSV generators from pbtxt/reference graphs such as wd_dcn or wide_and_deep.
---

# SRA Benchmark Model Builder

## Purpose

Build benchmark-only TensorFlow 1.x SavedModels and TSV inference data for predictor-server serving tests under `/home/c00913906/sra_benchmark`. The priority is to match the reference graph's inference signature, op mix, tensor shapes, and serving behavior for hardware performance evaluation. Model accuracy is secondary unless the user explicitly asks otherwise.

## Required Context

- Read `/home/c00913906/AGENTS.md` before changing model code or running training.
- If the target follows the completed `wd_dcn` work, read `references/wd_dcn-pattern.md`.
- Keep repository work inside `/home/c00913906/sra_benchmark` unless the user explicitly expands scope.
- Use `/home/c00913906/tmp` on the host and `/workspace/tmp` inside containers. Avoid `/tmp` because the root filesystem can be full.
- Use the `benchmark-train` container for model training and SavedModel validation unless the user names a different runtime.
- After SavedModel validation, use the active `benchmark_infer_workspace` container when available and collect one predictor-server timeline sample. Follow `tensorflow/BUILD_GUIDE.md` for the current serving binary names and client flags.
- The inference container mounts `/home/c00913906` as `/workspace`; the training container mounts `/home/c00913906/sra_benchmark` as `/workspace`.
- Use `/home/c00913906/models` as the default downloaded model artifact root and `/home/c00913906/dataset` as the default downloaded inference dataset root. In `benchmark_infer_workspace` these resolve as `/workspace/models` and `/workspace/dataset`.
- Run AtomGit download/upload on the host or in the inference container only; do not run AtomGit in `benchmark-train` because Python 3.7 and AtomGit compatibility is unreliable there.
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
6. Train inside `benchmark-train`, writing outputs under `/workspace/modelzoo/<model_name>/result/<version>`.
7. Validate the result: load the SavedModel, smoke-test inference, inspect the serving signature, and count important ops from the SavedModel graph.
8. If `benchmark-infer:v1.0.0` exists locally, validate through the active `benchmark_infer_workspace` runtime: run `predictor_server` against the SavedModel, run `brpc_client` against the generated TSV, and collect one timeline sample.
9. If an artifact already exists remotely, download it with `scripts/atomgit_artifacts.py` by `--model-name` before rebuilding.
10. Update `/home/c00913906/AGENTS.md` with the model path, TSV path, exact training command, concrete input feature inventory, SavedModel validation, predictor-server validation, and timeline path.
11. Upload completed model and dataset artifacts to AtomGit with `scripts/atomgit_artifacts.py` when the user asks to publish them.


## AtomGit Artifacts

Remote artifact repositories:

- Models: `codesheepchen/benchmark`
- Inference datasets: `codesheepchen/benchmark_dataset`

Default local roots:

- Host model root: `/home/c00913906/models`
- Host dataset root: `/home/c00913906/dataset`
- Predictor-container model root: `/workspace/models`
- Predictor-container dataset root: `/workspace/dataset`

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

The helper downloads the repository into the root and reports the expected path for the requested `model_name`. If the repository layout differs, pass `--source` for uploads and record the actual resolved path in `AGENTS.md`.

For `tools/atomgit_model_sync.py`, upload is additive at the remote repository level. It clears and recreates the local parts directory, but it does not delete stale remote split parts or old differently named artifacts. If a new archive has fewer parts, old extra remote parts remain even though the new manifest does not reference them.

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

## wd_dcn Construction Lessons

Use these rules when building or modifying `wd_dcn` or a structurally similar wide/deep/cross model.

### Preserve The Reference Slot Layout

- Read the reference graph as a slot layout, not as one embedding vector per feature family.
- The current `wide_and_deep.pbtxt` contract has 139 deep/trival slots and 41 wide slots, each with embedding dimension 12.
- Flattened first-layer widths are therefore `139 * 12 = 1668` for deep and `41 * 12 = 492` for wide.
- Do not average all `t` IDs into one `[batch, 12]` vector and then zero-pad it to 1668. Do not do the equivalent for `w`. That changes feature positions, nonzero counts, and the input contract of `SparseTensorDenseMatMul`.
- Preserve each populated ID's slot index, perform slot-level embedding lookup, and flatten the slot embeddings with `SparseReshape`.
- Validate `trival_count <= deep_slot_count` and `wide_count <= wide_slot_count`.

### Build Sparse Slot Embeddings Without A Large Table Concat

The reference graph uses custom row-fill and sparse-reshape operations. A standard-TensorFlow equivalent may use this sequence:

1. Flatten sparse `(batch, slot)` indices to rows with `batch_index * slot_count + slot_index`.
2. Add one extra segment at row `batch_size * slot_count` using an existing valid embedding ID such as 0.
3. Set the temporary sparse dense shape to `[batch_size * slot_count + 1, 1]`.
4. Run `tf.nn.embedding_lookup_sparse`.
5. Slice off the extra final output row.
6. Convert the remaining `[batch_size * slot_count, embedding_dim]` values to a SparseTensor.
7. Use `tf.sparse_reshape` to produce `[batch_size, slot_count * embedding_dim]`.

The extra segment forces `SparseSegmentMean` to materialize all required rows. Never append a zero row with `tf.concat([embedding_table, zero_row])`: for a multi-million-row table this introduces a large runtime `Concat` and table-sized memory movement.

After export, inspect every `Concat`/`ConcatV2` and assert that no embedding table is one of its inputs.

### Let Input Counts Define Initial Sparsity

- Do not add an artificial keep-ratio mask when the target sparsity is naturally represented by missing slots.
- Train with the standard full layout when required: `trival_count=139`, `wide_count=41`.
- Generate inference records independently from the training count. The current benchmark profile uses `t=49`, `w=41`, and `p=8`.
- For fixed-width slots, sparsity is `1 - populated_slot_count / total_slot_count`. Thus `t=49` over 139 slots is about `64.7%` sparse, while `w=41` over 41 slots is `0%` sparse.
- Each populated slot contributes up to 12 nonzero embedding values, so t=49 produces up to 588 nonzeros in 1668 dimensions and w=41 produces up to 492 nonzeros in 492 dimensions.
- Distinguish input-layout sparsity from downstream sparsity. The wide cross layers contain biases and can make previously zero positions nonzero before the wide MLP.

### Keep Training Shape And Inference Pressure Separate

- Model slot counts and MLP widths define graph structure.
- Per-record t/w counts define how many slots are populated.
- Client RPC batch size defines how many records are grouped into one request.
- Do not use a predictor-server `--batch_size` argument for client batching. Set client `--request_min_batch_size` and `--request_max_batch_size` instead.
- When reporting a timeline, state both the per-record t/w counts and the client-grouped RPC batch size.

### wd_dcn Validation Contract

For the current aligned graph, verify all of the following:

- Signature: one string input `model_input:0` with shape `[-1]`; score output shape `[-1, 1]`.
- Embedding variables use the configured bucket rows and dimension 12.
- First sparse weights are `[1668, 400]` and `[492, 400]`.
- `SparseSegmentMean=3`, `SparseReshape=2`, `SparseToDense=2`, `SparseTensorDenseMatMul=2`, `BatchMatMulV2=3`, `MatMul=7`, and `BiasAdd=0`.
- A batch-40 run with t=49/w=41 reshapes to `[40,1668]` and `[40,492]`.
- Deep has 588 nonzeros per sample and about `64.7%` sparsity; wide has 492 nonzeros per sample and `0%` sparsity before cross.
- No embedding table feeds a `Concat`/`ConcatV2`.
- Smoke-test both the intended inference counts and the full-slot boundary case.

When calculating embedding table storage, compute every row explicitly and verify the sum. For float32, byte size is `parameter_count * 4`; compare the expected result with the SavedModel variable file size.

## Validation Commands

Use container commands with an explicit temp directory, for example:

```bash
docker exec benchmark-train bash -lc 'cd /workspace/modelzoo/<model_name> && TMPDIR=/workspace/tmp python train.py --output_dir=./result/<version> ...'
```

Smoke-test a SavedModel from outside its source tree:

```bash
docker exec benchmark-train bash -lc 'cd /workspace && TMPDIR=/workspace/tmp python - <<"PY"
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

Every completed model build must list the concrete input feature inventory in the final response and in `/home/c00913906/AGENTS.md`. Include one row or bullet per feature family, for example:

- `g`: global scalar features, float, shape `[batch, 1]`, consumed by global branch.
- `t`: deep/trival id list, integer ids, count `128`, bucket `600000`, embedding dim `1668`, consumed by deep branch.

Use the model's actual reference-derived feature families. The `g/t/w/p` example is only for `wd_dcn`-style models.

## Predictor-Server Validation

After SavedModel smoke validation, check whether the benchmark inference image is available:

```bash
docker image inspect benchmark-infer:v1.0.0 >/dev/null 2>&1
```

If the image is available, prefer the existing `benchmark_infer_workspace` container. It mounts `/home/c00913906` as `/workspace`. Only start a temporary container when the active container is unavailable, following `tensorflow/BUILD_GUIDE.md`.

```bash
docker run -d \
    --name benchmark_infer_workspace_tmp \
    --cpuset-cpus=0-79 \
    --cpuset-mems=0 \
    --network=host \
    -v /home/c00913906:/workspace \
    benchmark-infer:v1.0.0 \
    tail -f /dev/null
```

Verify the image before reusing a container; do not infer the image from the container name.

Build the current server/client targets inside the inference container if needed:

```bash
cd /workspace/tensorflow
bazel --output_user_root=./output build --enable_bzlmod=true --experimental_enable_bzlmod --experimental_repo_remote_exec --cxxopt=-std=c++17 --host_cxxopt=-std=c++17 --copt=-O3 --host_copt=-O3 --copt=-march=armv8.5-a --host_copt=-I/usr/local/include --linkopt=-L/usr/local/lib64 --host_linkopt=-L/usr/local/lib64 --linkopt=-Wl,-rpath,/usr/local/lib64 --host_linkopt=-Wl,-rpath,/usr/local/lib64 --action_env=LD_LIBRARY_PATH=/usr/local/lib64:${LD_LIBRARY_PATH:-} --action_env=TF_SYSTEM_LIBS="boringssl,snappy" --distdir=/data/download --check_direct_dependencies=off //:predictor_server
bazel --output_user_root=./output build --enable_bzlmod=true --experimental_enable_bzlmod --experimental_repo_remote_exec --cxxopt=-std=c++17 --host_cxxopt=-std=c++17 --copt=-O3 --host_copt=-O3 --copt=-march=armv8.5-a --host_copt=-I/usr/local/include --linkopt=-L/usr/local/lib64 --host_linkopt=-L/usr/local/lib64 --linkopt=-Wl,-rpath,/usr/local/lib64 --host_linkopt=-Wl,-rpath,/usr/local/lib64 --action_env=LD_LIBRARY_PATH=/usr/local/lib64:${LD_LIBRARY_PATH:-} --action_env=TF_SYSTEM_LIBS="boringssl,snappy" --distdir=/data/download --check_direct_dependencies=off //:brpc_client
```

Inside the inference container, use paths under `/workspace`:

```bash
cd /workspace/tensorflow
TF_NUM_INTEROP_THREADS=16 TF_NUM_INTRAOP_THREADS=16 numactl -C 0-15 \
  ./bazel-bin/predictor_server \
  --enable_kdnn=false \
  --model_path=/workspace/sra_benchmark/modelzoo/<model_name>/result/<version>/saved_model/1 \
  --thread_num=16
```

From the same inference container, run the client against the generated TSV:

```bash
cd /workspace/tensorflow
./bazel-bin/brpc_client \
  --server=127.0.0.1:8000 \
  --input_data_path=/workspace/dataset/<model_name>.tsv \
  --thread_num=16 \
  --test_duration_s=10 \
  --warmup_duration_s=5 \
  --max_qps=1000 \
  --max_inflight=100 \
  --request_min_batch_size=1 \
  --request_max_batch_size=100
```

`brpc_client --max_qps` is global RPC QPS, not per-thread QPS; `--max_inflight` limits concurrent in-flight RPCs and overflow is counted as dropped/failure. `--request_min_batch_size` and `--request_max_batch_size` control how many samples are wrapped into each RPC.

Then collect one TensorFlow timeline sample by restarting the server with timeline enabled:

```bash
mkdir -p /workspace/timeline/<model_name>/<version>
cd /workspace/tensorflow
TF_NUM_INTEROP_THREADS=16 TF_NUM_INTRAOP_THREADS=16 numactl -C 0-15 \
  ./bazel-bin/predictor_server \
  --enable_kdnn=false \
  --model_path=/workspace/sra_benchmark/modelzoo/<model_name>/result/<version>/saved_model/1 \
  --thread_num=16 \
  --enable_tf_timeline=true \
  --tf_timeline_every_n=100 \
  --tf_timeline_max_dumps=1 \
  --tf_timeline_dump_warmup=false \
  --tf_timeline_dir=/workspace/timeline/<model_name>/<version>
```

Run the client once more while the timeline-enabled server is running. Record in `AGENTS.md`:

- The inference container name and image.
- The exact server and client commands.
- Client success or failure and any observed QPS/latency output.
- Timeline output directory and generated `*.runmeta.pb` files.

Convert a collected RunMetadata protobuf to TensorFlow Chrome trace JSON with the script documented in `tensorflow/BUILD_GUIDE.md`:

```bash
python /workspace/tools/runmetadata_to_timeline.py /workspace/runmeta_convert/<file>.runmeta.pb
```
