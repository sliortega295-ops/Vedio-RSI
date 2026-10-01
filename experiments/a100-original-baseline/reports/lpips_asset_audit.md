# Bounded read-only LPIPS asset audit (10:28 UTC)

The primary conversation authorized inspection of **only** an identified older LPIPS evaluation-asset directory, not its candidate code, reports, search history, timings or result artifacts. No old optimization material was opened or supplied to the clean Agents. No asset was copied or downloaded.

`/home/lyy/Experiments/SolRolloutBench/20260901-v0/assets/lpips/` contains only:

| Read-only asset | Size | SHA256 | Role |
|---|---:|---|---|
| `torch_home/hub/checkpoints/alexnet-owt-7be5be79.pth` | 244,408,911 B | `7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02` | AlexNet backbone weights; not a Python LPIPS package. |
| `torch_home/hub/pyiqa/LPIPS_v0.1_alex-df73285e.pth` | 6,009 B | `df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0` | pyiqa LPIPS linear weight asset; no permission to substitute pyiqa as the judge implementation. |

The archived `repo/tools/vision/lpips_judge.py` imports the **real** `lpips` module, constructs `lpips.LPIPS(net="alex", verbose=False)` (or without `verbose`), and calls `lpips.load_image` and `lpips.im2tensor` for frame pairs. Neither asset supplies that package/API. Bounded checks of `/usr/bin/python3`, `/home/lyy/.venv/bin/python3`, and the local Python 3.10 interpreter with installed Torch all returned `find_spec("lpips")=None`; the known local uv/pip cache has no `lpips` package directory, and the exact older evaluation-asset `wheelhouse-vbench-py312` contains no LPIPS wheel. The filenames alone suggest an LPIPS-v0.1-related asset but do **not** prove package compatibility or a passing metric run.

Thus read-only weights alone cannot satisfy the archived judge. The Cache aligned-LPIPS gate remains **BLOCKED** under the no-download rule. Existing pyiqa is not an interchangeable metric and was not used for promotion. If another explicitly approved, source-identifiable offline `lpips` package becomes available, its version/hash and the original judge API must be validated on the frozen aligned pairs before changing this status. The run currently has no such PASS.
