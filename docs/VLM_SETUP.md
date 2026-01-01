# Optional: running with a real VLM

**Nothing in `demo/` needs any of this.** `demo/quickstart.py` and
`demo/mini_benchmark.py` answer the discrete variables `Φ` with the simulator
oracle (`pipeline/oracle.py`), which is also what the paper's large-N
campaigns do — it isolates the continuous stage and costs nothing. You only
need a VLM to reproduce the real-VLM campaigns (D, I, J, K, L, M) or to run
the method the way it would be deployed.

The interface is a **local, OpenAI-compatible endpoint**. Every consumer
(`alkbench/discrete.py::VLMSolver`, `baselines/moka_style.py`,
`baselines/moka_marks.py`, `baselines/rekep_real.py`,
`pipeline/campaign_i.py`) reads the same three environment variables and
imports `openai` **lazily**, so the package installs and the tests pass
without it:

```bash
export ALK_VLM_BASE_URL=http://127.0.0.1:8199/v1
export ALK_VLM_API_KEY=local          # any non-empty string; vLLM ignores it
export ALK_VLM_MODEL=Qwen/Qwen3-VL-32B-Instruct-FP8
pip install -e '.[vlm]'               # openai client + torch
```

There is no hosted API in the loop and no credential to manage: the endpoint
is a vLLM server you run yourself.

## The two model families

The campaigns use two deliberately unrelated vision-language models, so that
a finding can be attributed to the *question* rather than to one model.

| role | model | notes |
|---|---|---|
| primary (all real-VLM campaigns: D, I, J, L) | `Qwen/Qwen3-VL-32B-Instruct-FP8` | official Qwen FP8 release, ≈35.5 GB of weights |
| second family (campaign M) | `RedHatAI/gemma-3-27b-it-FP8-dynamic` | Gemma 3 text decoder + SigLIP-so400m vision tower, FP8-dynamic (W8A8) — the *same quantisation class* as the Qwen model, so the comparison is not confounded by precision |

The second family was chosen because it shares nothing with the first:
different pretraining data, tokenizer, chat template and vision encoder, and a
language model from a different lineage. (InternVL3-38B/14B would *not*
qualify — their language models are Qwen2.5 checkpoints.)

## Serving with vLLM

```bash
python -m venv .venv-vllm && .venv-vllm/bin/pip install vllm
.venv-vllm/bin/vllm serve Qwen/Qwen3-VL-32B-Instruct-FP8 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.6 \
    --port 8199
```

`--gpu-memory-utilization 0.6` is not arbitrary: MuJoCo rendering for the
evaluation runs on the same GPU and needs the headroom. If you serve on a
separate machine, raise it.

Health check: `curl http://127.0.0.1:8199/v1/models`

### Consumer-Blackwell (sm_120) workarounds

On an RTX PRO 6000 / consumer Blackwell card, two vLLM subsystems fail their
arch checks. Export both and the server comes up on the default wheel
(vllm 0.27.1, torch cu130 — `sm_120` is in the default wheel's arch list, so
no source build is needed):

```bash
export VLLM_USE_DEEP_GEMM=0            # DeepGEMM FP8 kernels crash on sm_120
                                       # ("Unknown SF transformation");
                                       # the CUTLASS/Marlin fallback is fine
export VLLM_USE_FLASHINFER_SAMPLER=0   # FlashInfer's JIT sampler fails its
                                       # arch check ("requires sm75 or
                                       # higher"); the torch sampler is fine
```

Serving from a local weights directory (useful when the Hugging Face python
client stalls on a slow link — `curl -C -` resumes, the HF client's temp files
do not) still keeps the public model id in every record:

```bash
.venv-vllm/bin/vllm serve /path/to/local/weights \
    --served-model-name RedHatAI/gemma-3-27b-it-FP8-dynamic \
    --max-model-len 16384 --gpu-memory-utilization 0.6 --port 8199
```

Pin the revision when you fetch weights, so the `model` field in the records
identifies an exact commit.

## Checking that it works

```python
from alkbench.discrete import VLMSolver
solver = VLMSolver("Qwen/Qwen3-VL-32B-Instruct-FP8")   # reads the env vars
```

`VLMSolver` presents a numbered-marker image (rendered by
`draw_candidate_markup` + `encode_png` — pure numpy and zlib, no PIL or
OpenCV), asks for JSON-only answers with 1-based candidate indices, validates
the reply syntactically and retries once with the error fed back. This is
the prompt the reported campaigns used; see `alkbench/NOTES.md` item 8.

The live-VLM test is skipped unless you opt in:

```bash
ALK_VLM_TEST=1 python -m pytest tests/test_baseline_moka.py -q
```

## torch, and why it is optional

The only torch dependency in the whole repository is the ReKep baseline's
DINOv2 keypoint proposal (`baselines/rekep_keypoint_proposal.py`, ViT-S/14 via
`torch.hub`). It runs as a **separate interpreter process** so that torch
never has to be installed alongside the simulator:

```bash
export ALK_TORCH_PYTHON=/path/to/an/interpreter/with/torch
```

The default is the current interpreter, which works if you installed the
`[vlm]` extra. The worker runs with `CUDA_VISIBLE_DEVICES=""` — on CPU — on
purpose, because the GPU is busy serving the VLM.

## Cost discipline

Every real-VLM driver **caches every query and reply on disk** and skips
anything already answered, so a re-run costs nothing and a crash costs only
the queries in flight. Prompts and decoding parameters are selected in a
`freeze` phase on **non-evaluation scenes** (easy tier, seeds 1–5, and the
demonstration scenes) and written to a `freeze.json`; nothing after the freeze
is tuned. Keep that discipline if you add an arm — it is what makes the
comparison honest.
