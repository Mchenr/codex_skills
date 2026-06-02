# wd_dcn Reference Pattern

## Completed Workload Contract

The completed `wd_dcn` workload is the reference implementation for future SRA benchmark model builds.

- Repo path: `/home/c00913906/sra_benchmark/modelzoo/wd_dcn`
- Predictor-server path: `/data/sra_benchmark/modelzoo/wd_dcn`
- Reference SavedModel in repo workspace: `/data/sra_benchmark/modelzoo/wd_dcn/result/version_b_graphopt_sparse0065_emb12x/saved_model/1`
- Preferred downloaded model artifact root: `/data/models/wd_dcn`
- Reference inference TSV in repo workspace: `/data/sra_benchmark/modelzoo/wd_dcn/result/version_b_infer.tsv`
- Preferred downloaded dataset artifact root: `/data/dataset/wd_dcn`
- Serving input schema: `id<TAB>model_input`
- `model_input` format: `g=<float>;t=<deep ids>;w=<wide ids>;p=<plain ids>`

## AtomGit Artifact Commands

Download all published benchmark models and resolve the expected `wd_dcn` path:

```bash
python /home/c00913906/.codex/skills/sra-benchmark-model-builder/scripts/atomgit_artifacts.py \
  download-model --model-name wd_dcn
```

Download all published inference datasets and resolve the expected `wd_dcn` path:

```bash
python /home/c00913906/.codex/skills/sra-benchmark-model-builder/scripts/atomgit_artifacts.py \
  download-dataset --model-name wd_dcn
```

Upload a completed model version:

```bash
python /home/c00913906/.codex/skills/sra-benchmark-model-builder/scripts/atomgit_artifacts.py \
  upload-model --model-name wd_dcn \
  --source /home/c00913906/models/wd_dcn
```

Upload inference data:

```bash
python /home/c00913906/.codex/skills/sra-benchmark-model-builder/scripts/atomgit_artifacts.py \
  upload-dataset --model-name wd_dcn \
  --source /home/c00913906/dataset/wd_dcn
```

Run these AtomGit commands on the host or in the predictor container, not in `benchmark-train-dev`; Python 3.7 and AtomGit are incompatible in the training container. Do not place credentials in this file. Use the local AtomGit CLI login state or a shell environment variable.

## Graph Shape

The graph is a benchmark approximation of the reference `wide_and_deep.pbtxt`, not a product model.

- Deep/trival features are parsed from `t=` IDs, embedded, sparsified, and passed through the deep branch.
- Wide features are parsed from `w=` IDs, embedded, crossed by DCN-style cross layers, and passed through the wide MLP branch.
- Plain features are parsed from `p=` IDs and embedded into a compact plain branch.
- Global scalar features are parsed from `g=` and contribute a small global branch.
- Final logits are produced by summing branch logits and applying sigmoid for serving output.

## Op Expectations

For the current serving target, preserve these expectations unless the user changes the reference graph:

- `SparseTensorDenseMatMul`: exactly 2 in the clean SavedModel.
- The two sparse dense matmuls are used by the first deep dense layer and the first wide MLP dense layer.
- Later MLP layers use dense `MatMul`.
- `BiasAdd`: 0 in the serving graph.
- Biases should be represented through explicit add operations.
- Batch-norm or equivalent scale/offset state should be represented as serving variables where required by the reference behavior.

## Training Command Template

Use the completed command shape as the starting point:

```bash
cd /workspace/modelzoo/wd_dcn
TMPDIR=/workspace/tmp python train.py \
  --output_dir=./result/wd_dcn \
  --steps=200 \
  --batch_size=128 \
  --synthetic_records=200000 \
  --seed=2026 \
  --trival_count=128 \
  --wide_count=64 \
  --plain_count=8 \
  --deep_bucket_size=600000 \
  --wide_bucket_size=600000 \
  --plain_bucket_size=262145 \
  --deep_embedding_dim=1668 \
  --deep_input_keep_ratio=0.065 \
  --deep_input_sparsity_mode=deterministic_random \
  --deep_input_sparsity_seed=2026 \
  --wide_embedding_dim=492 \
  --plain_embedding_dim=5 \
  --cross_layers=3 \
  --deep_hidden_units=400,400,400 \
  --wide_hidden_units=400,400,400 \
  --clean_output=True
```

For other models, keep the command deterministic and store the exact final command in `AGENT.md`.

## Inference Data

Generate TSV inference data with `tools/gen_infer_data.py` or the model-specific equivalent. Keep the generated features within the same bucket and feature-count assumptions used by the exported graph.

Expected output shape:

```text
id	model_input
0	g=0.123;t=1,2,...;w=3,4,...;p=5,6,...
```

## AGENT.md Update Checklist

After a model is built, record:

- Model name and version directory.
- Host path and predictor-server path for `saved_model/1`.
- Host path and predictor-server path for inference TSV.
- Exact training command.
- Exact inference-data generation command.
- Signature and important op counts.
- Any deviations from the reference graph and why they are acceptable for benchmark use.
