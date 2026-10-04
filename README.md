# DeepSeek-V4.1-Flash on 2× DGX Spark (GB10) — our TensorFold TP2 engine

**BertholomusAI recipe: DeepSeek-V4.1-Flash served tensor-parallel over two NVIDIA DGX Spark / GB10 nodes by a
`deepseek_v41` model family we wrote for [TensorFold](https://github.com/ashhart/TensorFold) (by ashhart).**

> **Weights:** [Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw)
> (MIT), an EXL3 quant by Mia-AiLab of [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
> (MIT, © 2023 DeepSeek). We did not make or modify these weights; this repository holds no weights. The Engram tables
> are read at run time from DeepSeek's original shards 47 and 48.

**Engine:** [bertholomus/TensorFold, branch `deepseek-v41-tp2`](https://github.com/bertholomus/TensorFold/tree/deepseek-v41-tp2).
Design, full report and attribution: [`tools/dsv41/`](https://github.com/bertholomus/TensorFold/tree/deepseek-v41-tp2/tools/dsv41).
Clean-room: the model math is re-implemented from DeepSeek's MIT inference code and tech report; no code from other
DeepSeek-V4.1 recipes or kits was read or copied ([ATTRIBUTION.md](https://github.com/bertholomus/TensorFold/blob/deepseek-v41-tp2/tools/dsv41/ATTRIBUTION.md)).

## Measured (release build, 2026-10-04, 2× GB10, TP2)

Same public client (`tools/dsv41/kit_bench.py`), needle, vision and concurrency tools as in the report. Greedy decode,
DSpark drafting with depth from the confidence head.

- **Single stream, 512 tokens:** code / prose / structured **60.5 / 38.0 / 74.5 tok/s**
  (384-token prompt set b: 69.9 / 42.3 / 96.5)
- **4 concurrent streams:** **73.2-80.5 tok/s aggregate**; every concurrent reply bit-identical to its solo reply
  (12/12 burst, 12/12 staggered, greedy and seeded T=0.6). 2 streams: 53.7-58.0.
- **Cold prefill (bounded replay):** 8K 1,349-1,365 · 32K 1,247 · 64K 1,342 · 128K 1,262 tok/s
- **Decode after a 128K prompt:** 61.8 tok/s
- **Needles:** 12/12 at 8K / 32K / 128K / 250K × 3 depths. Window up to 1,048,576 tokens (packed FP4 KV); an earlier
  build found a needle at 1,039,833 tokens (not re-run on the release build).
- **Vision:** native image input, 6/6 checks (colour, text, two images, image + tool call, image in a tool result,
  refusal in an assistant turn)
- **Drafted == serial:** 8/8 at T=0; greedy output identical run to run
- **Start to ready:** 65 s with the per-rank weight cache (first start that writes it: 330 s)
- **Kept prompts:** a 17.9K-token agent prompt with 24 tool schemas: first token 12.7-19.9 s → 1.2-2.4 s, replies equal
  to fresh prefill 10/10

## Run it

Two GB10 nodes with a direct RoCE link. On each node, a throwaway `nvcr.io/nvidia/pytorch:26.07-py3` container with
the `deepseek-v41-tp2` branch installed (`pip install -e`). Start rank 1 (worker) first, then rank 0 (head):

```
TF_DS_REPLAY=1 TF_DS_PREFILL_CHUNK=2048 TF_DS_RANK_CACHE=<CACHE_DIR> tensorfold serve <MODEL_DIR> \
  --tp 2 --rank R --master <HEAD_IP> --host 127.0.0.1 --port 18891 --context 262144 --vision --parallel 4 \
  --mtp-drafts 5 --temperature 0
```

- `<MODEL_DIR>`: the Mia-AiLab EXL3 checkpoint. Engram tables: `TF_DS_ENGRAM=<ENGRAM_DIR>` (DeepSeek's original shards
  47/48), or a folder next to `<MODEL_DIR>` whose name contains "Engram".
- `NCCL_SOCKET_IFNAME` / `NCCL_IB_HCA` / `NCCL_IB_GID_INDEX` select the RoCE link (GID index 5 by default).
- `TF_DS_RANK_CACHE` keeps each rank's weights in one file (~106 GB a rank, written on the first start).
- `--context` up to 1048576. `TF_API_KEY_FILE=<file>` makes every route but `/health` require a key.
- Benchmark: `python3 tools/dsv41/kit_bench.py --base http://127.0.0.1:18891 --model <name> decode|concurrent|prefill|depth`

Hosts and addresses are placeholders; substitute your own.

## Credits

- **DeepSeek** — DeepSeek-V4.1-Flash, its inference code and tech report (MIT, © 2023 DeepSeek).
- **Mia-AiLab** — the EXL3 2.9 bpw quant this recipe serves (MIT).
- **ashhart** — [TensorFold](https://github.com/ashhart/TensorFold) (Apache-2.0), the engine this family plugs into.
- **turboderp** — [EXL3 / exllamav3](https://github.com/turboderp-org/exllamav3) (MIT), the weight format.
- **BertholomusAI** (Albert Lee, [bertholomus](https://github.com/bertholomus)) — the `deepseek_v41` TP2 family, its
  kernels, the deployment and the measurements.

Not affiliated with or endorsed by DeepSeek, NVIDIA, the TensorFold authors, Mia-AiLab or MiaAI-Lab.

## License

Recipe documentation: Apache-2.0 (`LICENSE`). The weights stay under their own MIT license.
