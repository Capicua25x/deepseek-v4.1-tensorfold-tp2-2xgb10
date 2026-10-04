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

## v0.2 (2026-10-04): what changed, measured against v0.1.0

Same nodes, same public client (`tools/dsv41/kit_bench.py`), same prompts and same method as v0.1.0. Greedy decode,
DSpark drafting, served configuration below. Median of 3 unless noted.

- **Single stream, 384 tokens (prompt set b):** code **69.9 → 80.4 tok/s (+15%)**, prose 42.3 → 49.2 (+16%),
  structured **96.5 → 111.1 (+15%)**
- **Single stream, 512 tokens:** code 60.5 → 70.8 (+17%), prose 38.0 → 44.2 (+16%), structured 74.5 → 84.6 (+14%)
- **4 concurrent streams, 384 tokens, total tokens / wall clock:** **80.5 → 91–93.5 tok/s (+13–16%)** (medians of four
  separate runs of 3–9 reps; this row varies with the server start more than the others); 256 tokens 77.5 → 85.6 (+10%)
- **4 streams sustained (new measure):** 4 requests always in flight for 90 s, distinct prompts: **103 tok/s aggregate**,
  per-stream p50 31 tok/s, first token p50 0.22 s
- **Prompt speed (bounded replay):** 8K **1,357 → 1,935 tok/s (+43%)**, 32K 1,247 → 1,953 (+57%), 64K 1,342 → 1,897 (+41%),
  128K 1,262 → 1,741 (+38%)
- **Start to ready** (restart with the per-rank weight cache, first reply included): **65 s → 35 s (−46%)**
- **Quality (new):** fixed, seeded subsets, greedy: MMLU-200 87.5% (thinking off) / 89.0% (on, 4,096-token cap);
  GSM8K-100 97.0% / 96.0%. Script: `tools/dsv41/quality_eval.py` (subset digest 09c2e6cc43e275af).
- **Still exact:** drafted == serial 8/8 at T=0; every concurrent reply bit-identical to its solo reply (12/12 burst,
  12/12 staggered); solo replies identical to v0.1.0's (12/12: the new kernels are bit-identical); images 6/6;
  needles 12/12 at 8K–250K.

Where it came from: grouped EXL3 decode linears (884 → 350 launches a forward) and a fused grouped-expert decode path,
both bit-identical; Engram reads moved off the critical path; a lighter per-round rank check; NCCL over both RoCE ports
for prompt chunks; a faster weight-cache read and a shorter warm-up. Details in `tools/dsv41/REPORT.md`.

## v0.1.0 (2026-10-04, first release)

Single stream 512 tokens code / prose / structured 60.5 / 38.0 / 74.5 tok/s (set b 69.9 / 42.3 / 96.5); 4 streams
73.2–80.5 tok/s; prefill 1,247–1,365 tok/s at 8K–128K; decode after a 128K prompt 61.8 tok/s; window up to 1,048,576
tokens (an earlier build found a needle at 1,039,833 tokens); start to ready 65 s.

## Run it

Two GB10 nodes with a direct RoCE link. On each node, a throwaway `nvcr.io/nvidia/pytorch:26.07-py3` container with
the `deepseek-v41-tp2` branch installed (`pip install -e`). Start rank 1 (worker) first, then rank 0 (head):

```
TF_DS_REPLAY=1 TF_DS_PREFILL_CHUNK=2048 TF_DS_RANK_CACHE=<CACHE_DIR> TF_DS_RANK_CACHE_READERS=32 \
TF_DS_WARM_LENGTHS=1,17,33,131,514,1024,2113 \
NCCL_IB_HCA=<HCA_PORT_0>,<HCA_PORT_1> NCCL_IB_GID_INDEX=5 NCCL_SOCKET_IFNAME=<IFACE> \
tensorfold serve <MODEL_DIR> --tp 2 --rank R --master <HEAD_IP> --host 127.0.0.1 --port 18891 \
  --context 262144 --vision --parallel 4 --mtp-drafts 5 --temperature 0
```

- `<MODEL_DIR>`: the Mia-AiLab EXL3 checkpoint. Engram tables: `TF_DS_ENGRAM=<ENGRAM_DIR>` (DeepSeek's original shards
  47/48), or a folder next to `<MODEL_DIR>` whose name contains "Engram".
- `NCCL_IB_HCA` lists both RoCE ports when both are cabled (prompt-chunk gathers 153 → 106 ms); one port works too.
- `TF_DS_RANK_CACHE` keeps each rank's weights in one file (~106 GB a rank, written on the first start).
- `--context` up to 1048576. `TF_API_KEY_FILE=<file>` makes every route but `/health` require a key.
- Benchmark: `python3 tools/dsv41/kit_bench.py --base http://127.0.0.1:18891 --model <name> decode|concurrent|sustained|prefill|depth`
- Quality: `python3 tools/dsv41/quality_eval.py --base http://127.0.0.1:18891 --model <name> --mmlu <MMLU_TEST_PARQUET> \
  --gsm8k <GSM8K_TEST_JSONL> --max-tokens-on 4096` (MMLU "all" test from cais/mmlu, GSM8K test from openai/grade-school-math)

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
