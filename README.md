# RBLN Prefill/Decode Disaggregation kits

Deployable kits for **prefill/decode (PD) disaggregation** of LLM serving on **Rebellions RBLN** NPUs
(the `vllm_rbln` optimum path), including a **heterogeneous** variant that prefills on a GPU and
decodes on the NPU. Model-agnostic — you choose the model, tensor-parallel size, and batching.

Everything shippable is in **`dist/`** as three self-contained tarballs. Each unpacks with a
`README.md` and a step-by-step operator `RUNBOOK.md`.

## Why disaggregate
On the RBLN optimum path, prefill runs one request per engine step, so a new request's prefill stalls
in-flight decodes. Moving prefill off the decode engine removes that stall — either to other NPU cards
(all-RBLN PD) or to a GPU (heterogeneous PD, for parallel prefill).

## The kits (`dist/`)
| tarball | role |
|---|---|
| `pd_disagg_consumer_npu.tar.gz` | RBLN NPU **decode pool** (N×TP decoders; receives KV over pd_nixl) — used by both producers |
| `pd_disagg_producer_npu.tar.gz` | RBLN NPU **prefill producer** + `epd_proxy.py` (all-RBLN PD) |
| `pd_disagg_producer_gpu.tar.gz` | GPU **prefill producer** + KV bridge (heterogeneous PD) |

Bring the **consumer up first** — the producer needs the decode box's IP and the decoders' pd_nixl
ports.

## The KV bridge (heterogeneous case)
RBLN stores KV in a 2-byte on-device float type with an RBLN-specific per-layer block layout, which is
bit-incompatible with a GPU's IEEE KV. The GPU producer therefore **recodes** the KV to the RBLN dtype
and **retiles** it to the RBLN block layout before publishing; RoPE is not re-applied (both sides cache
post-RoPE K with the model's own rotary convention — verify per model family) and TP head-sharding is
handled inside the RBLN write. The bridge code ships in `pd_disagg_producer_gpu.tar.gz/kv_bridge/`.

## Quick start
```bash
# on the NPU box
tar xzf dist/pd_disagg_consumer_npu.tar.gz && cd pd_disagg_consumer_npu   # follow RUNBOOK.md
# then a producer (NPU or GPU), pointed at the consumer — follow its RUNBOOK.md
```

## Requirements
- **NPU:** RBLN cards, `vllm_rbln` + `optimum-rbln` + `rebel-compiler` (apply the overlay via each
  kit's `install.sh`).
- **GPU producer:** a CUDA GPU, deps in `pd_disagg_producer_gpu.tar.gz/requirements-gpu.txt`.
