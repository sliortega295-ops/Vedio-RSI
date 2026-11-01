> 2026-10-31 发布副本：保留原报告的实验日期、结论和历史状态；私有路径已替换，未发布产物不伪装为可访问链接。文末“未推送”等措辞指报告生成时；本次发布说明见根目录 PUBLICATION-NOTE.md。

# Existing Wan assets audit

The existing local Diffusers layout is suitable as an original-weight baseline: all 1,261 serialized weight tensors exactly equal the official native checkpoint after key renaming, at the same stored dtype. This establishes weight identity, not generated-output parity or official Diffusers snapshot byte identity.

Audit completed 2026-10-24 16:05:52 UTC on job-node-a-db4b-4527-8eb8-7518319a0277-master-0. All remote checks were read-only and CPU-only. No model payload was downloaded, no model weights were transferred locally, and no original CoFrame metadata or implementation was changed.

## Original source and conversion

Native directory: <REMOTE_PROJECT_PATH>

Native official repository: Wan-AI/Wan2.1-T2V-1.3B, revision 37ec512624d61f7aa208f7ea8140a131f93afc9a. Fresh SHA256 and byte counts for the three weights and two tokenizer payloads match both the existing native manifest and the [live official pinned-revision LFS metadata](https://huggingface.co/api/models/Wan-AI/Wan2.1-T2V-1.3B/tree/37ec512624d61f7aa208f7ea8140a131f93afc9a?recursive=true&expand=false).

| Original native file | Bytes | Fresh SHA256 |
| --- | ---: | --- |
| diffusion_pytorch_model.safetensors | 5,676,070,424 | 96b6b242ca1c2f24e9d02cd6596066fab6d310e2d7538f33ae267cb18d957e8f |
| models_t5_umt5-xxl-enc-bf16.pth | 11,361,920,418 | 7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d |
| Wan2.1_VAE.pth | 507,609,880 | 38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981 |

Converted directory: <REMOTE_PROJECT_PATH>

Its metadata declares Wan-AI/Wan2.1-T2V-1.3B-Diffusers revision 0fad780a534b6463e45facd96134c9f345acfa5b. That declaration is separate from the verified native revision. The directory is a local conversion and must not be described as a byte copy of an official Diffusers snapshot.

The existing vendored conversion mapping, <REMOTE_PROJECT_PATH> has SHA256 330442aad93b365857ec57cee34b0f0380ab5f7e6be178a33b66bcbdc63c0e41, exactly matching the current [upstream Diffusers v0.34.0 converter source](https://raw.githubusercontent.com/huggingface/diffusers/v0.34.0/scripts/convert_wan_to_diffusers.py). Only its key mappings were used for this audit; no conversion pipeline was rerun.

## Complete CPU tensor comparison

| Component | Original keys | Converted keys | Exactly equal | Stored dtype | Elements compared |
| --- | ---: | ---: | ---: | --- | ---: |
| Transformer | 825 | 825 | 825 | FP32 → FP32 | 1,418,996,800 |
| Text encoder | 242 | 242 | 242 | BF16 → BF16 | 5,680,910,336 |
| VAE | 194 | 194 | 194 | FP32 → FP32 | 126,892,531 |

All components had zero mapping collisions, missing target keys, extra target keys, unused mapped source keys, mismatches, cast-only matches, or BF16-roundtrip-only matches. The original VAE checkpoint separately contained 194 keys. Transformer block indices cover 0 through 29; the native patch-embedding shape is [1536,16,1,2,2], and the first self-attention Q/K weight shapes are [1536,1536].

Native configuration confirms hidden dimension 1536, 12 heads and 30 layers. Converted configuration confirms head dimension 128, 12 heads, 30 layers, FFN dimension 8960, text dimension 4096, 16 input/output channels and patch size [1,2,2].

The current converted Transformer is stored in FP32. A later runtime cast to BF16 is an inference precision choice and must be recorded separately.

## Metadata discrepancies and tokenizer scope

The existing converted MODEL_PROVENANCE.json records native_text_encoder SHA256 d92de679881d38af9c89eff7bb1b6d6c9d96cb2b69831e4027e9ecabdd38eb23. The current official native file is 7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d. All 242 converted text-encoder tensors nevertheless match the current official native file exactly. The recorded field is stale or inconsistent; its historical cause was not established.

The existing local wrapper, <REMOTE_PROJECT_PATH> has SHA256 181f8a98ae56d14cdcdd7b19e923c06d7aad6b3bf23580db1520e6c76af279f4. It contains a Transformer BF16 cast, whereas the current converted Transformer is FP32. The wrapper was inspected but not executed; it must not be claimed as an exact reconstruction of current cache bytes.

The converted SentencePiece payload matches both official sources. Converted tokenizer.json has SHA256 20a46ac256746594ed7e1e3ef733b83fbc5a6f0922aa7480eda961743de080ef and matches the [declared official Diffusers revision metadata](https://huggingface.co/api/models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/tree/0fad780a534b6463e45facd96134c9f345acfa5b?recursive=true&expand=false). Compared with native tokenizer.json, its decoder and pre-tokenizer replace add_prefix_space=true with prepend_scheme="always" and split=true. Runtime tokenization parity was not tested. Current tokenizer_config.json is 61,728 bytes versus the declared official revision tree's 61,758 bytes; full tokenizer configuration byte identity is not claimed.

The 17 runtime model/configuration files are freshly fingerprinted in artifacts/asset-audit.json, totaling 17,567,067,207 bytes. Original MODEL_PROVENANCE.json and PINNED_REVISION are recorded separately as metadata_files, with their provenance claims explicitly not admitted as verified. The launcher should use model_files when staging the clean experiment copy.

## Installed runtime source check

Python: /tmp/distillschedule-helios-official-torch210/bin/python

Torch: 2.10.0+cu128; Diffusers: 0.38.0.dev0; CUDA_VISIBLE_DEVICES empty; CPU threads 4.

The actually imported Wan Transformer, Wan pipeline and UniPC scheduler sources match their installed RECORD hashes/sizes and the exact entries in <REMOTE_PROJECT_PATH> That wheel's fresh SHA256 is c90e20ddbdf626442d2d06f52560f7306453ba8608d67e67a3a192aca379ee21 and matches installed direct_url metadata.

| Imported source | Fresh SHA256 |
| --- | --- |
| diffusers/models/transformers/transformer_wan.py | 2540d7704e5bbd2f65002f32a26caf2a5215741075f3ba98c6527e58b1608c15 |
| diffusers/pipelines/wan/pipeline_wan.py | 20df2172342aeaaa6e4d70a966b538cfbf3fd153e8edbd67a549bbf2238868ac |
| diffusers/schedulers/scheduling_unipc_multistep.py | 0085cfa07ec827662eda5f196386add332aaf34f8954ad52380a070b35f9cd22 |

This verifies those installed sources against the existing wheel. The wheel's upstream Git revision and all other dependencies were not audited.

## Reproducibility and limits

scripts/audit_existing_wan.py is the exact Python body executed for layout inventory and complete tensor parity, SHA256 34257adead091b391844f556f88056bfa3a2d1cf45352621e1d94f31ee825fad. It extracts only conversion mappings, forces native PTH loads to CPU/mmap, and avoids model construction. Other provenance/hash checks are recorded in the JSON receipt and were separate read-only checks.

From this local run directory, the recorded execution is:

    ssh -p 2222 -o BatchMode=yes -o ConnectTimeout=20 \
      '<SSH_TARGET>' \
      'PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /tmp/distillschedule-helios-official-torch210/bin/python -B -' \
      < scripts/audit_existing_wan.py

Native-versus-Diffusers generated-output parity, tokenization runtime parity, inference quality, latency and GPU behavior are NOT_RUN here.

Recommended provenance wording: Official Wan-AI/Wan2.1-T2V-1.3B native checkpoint at revision 37ec512624d61f7aa208f7ea8140a131f93afc9a, represented locally in Diffusers layout using the v0.34.0 conversion key mappings. All 1,261 serialized weight tensors equal the original native checkpoint exactly at their native dtypes. This is a local conversion, not an official Diffusers snapshot byte copy.
