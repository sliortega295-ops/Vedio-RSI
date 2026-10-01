# AlexNet evaluation-weight acquisition — remote node only

The user had previously proposed a Hugging Face mirror as an alternative
route. From the authorized A100 node, a `HEAD -L` to the mirror's
`worldbench/videogen` AlexNet object returned final HTTP 200 and
`Content-Length: 244408911`; its Hugging Face listing advertised the same full
SHA256 as the official Torchvision checkpoint. This was a route preflight, not
a local download. The mirror copy was accepted only after independently
checking the *downloaded bytes* against that full digest.

- Local project script: `scripts/phase3_download_alexnet.sh`, SHA256
  `70b434c5d5da15716e09d3540b542e77cb2cf590f1ca842a84045b917c9a7138`;
  `bash -n` passed. A project-scoped, SHA-checked PTY transfer placed the small
  script on the A100 node (archive SHA256
  `a6d1b85548056f0f5566954892b8b10df2c7063d20d8e5e8874acc3f052b7f5b`).
- Remote download start: `2026-09-14T02:46:00Z`. `curl` reported final HTTP 200,
  `244408911` body bytes and `32.527195` seconds. End: `02:46:34Z`.
- The object was staged as a project-local `.partial` file; only after size and
  full SHA256 matched was it renamed to
  `deps/phase3/torch_home/hub/checkpoints/alexnet-owt-7be5be79.pth` on the
  **remote** node. A separate remote `stat`/`sha256sum` recheck found size
  `244408911` and SHA256
  `7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02`.
- No SANA model was downloaded or changed, no local AlexNet asset was uploaded,
  no weight bytes traversed the local SSH session, no persistent proxy setting
  was made, and the A100 node had no compute app at the post-download check.

This receipt establishes the weight's provenance and byte identity, **not**
LPIPS quality or Cache acceptance. The archived judge still needs a positive
network-denied self-pair test, followed by exact aligned five-prompt evaluation
and the original Master quality gate before any Cache promotion.
