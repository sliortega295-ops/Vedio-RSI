"""Host-only typed bridge to the trusted remote original Cache CPU judges.

This executes no Agent-owned script or shell text. It admits only exact,
coordinator-reviewed real seq0002 Cache runs with hash-pinned predecessors.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from orchestration import site_remote


PROJECT = Path("/home/lyy/Experiments/SolAgent/20260913-a100-original-baseline")
COORD = PROJECT / "repo"
CACHE = COORD / "output/experiments/sana2ba100-cache_ca-0002/worktree"
REMOTE_CACHE = site_remote.REMOTE_PROJECT / "repo/output/experiments/sana2ba100-cache_ca-0002/worktree"
LOCAL_WORKER = COORD / "orchestration/site_quality_worker.py"
REMOTE_WORKER = site_remote.REMOTE_PROJECT / "repo/orchestration/site_quality_worker.py"
TRUSTED_WORKER_SHA256 = "f7e5d3acdc9f1c82f7d4abbdb75a75aa99766a551e24a57353ce5ab5af22a3c7"
REMOTE_PYTHON = "/tmp/distillschedule-helios-official-torch210/bin/python3.10"
TRUSTED_PYTHON_SHA256 = "7c9e72ea1f06138ed5f52f49894c62f29e51e302f0d02f8fb18490e90e994331"
SUCCESS = "lpips_complete_visual_and_other_quality_gates_pending"
EXPECTED_BUNDLES = {
    "R01": {
        "name": "20260913-095257-cache_ca_r01_easycache_t010",
        "launch_sha256": "aa6ea3059070bef51850b435ade8242054364f87f3ded466d8347857d506f7f6",
        "manifest_sha256": "cc32f37612dd68f28819cb0e558c2aac0798652ab5c691f9ec2932ce2a1dd61a",
        "metadata_sha256": "c43a4442d5f9e98da87d2f623b140bc4dde9974037011d1a090761f6282140a5",
        "benchmark_sha256": "9e18f8a4a7f9be54886c7290d16d3e4a2eb86c67dc6b37d1a22420462aafe111",
        "run_config_sha256": "f55df5ed499e978867c2f0398028a6e73765d5fdd9df24adb15f7bb6439fda73",
        "quality_sha256": "a49d87a87ff9b98bf33be56aa149e2a68ea788953cfff596340061128225a197",
        "collection_sha256": "ea6db12c54ea2ab7d60e20d8ad1105e9e682e69075cdf567e11fb5b2d3b4370a",
        "assess_sha256": "d8767370d252a5f52b8b881f237a3429938d0fa338f5fea4cb026e1b364e9a43",
        "total_s": 375.18,
    },
    "R02": {
        "name": "20260913-102721-cache_ca_r02_teacache_t010",
        "launch_sha256": "cace2f725792095f889c6659a9189eca538cb5aee0b941c596da5a0986c283e9",
        "manifest_sha256": "6d857fc443247539ec113d2b32a6c01d85a6729e7d25ab68794379328facb135",
        "metadata_sha256": "72bfa2a22637fe441eeb0f1fc2c9dba89d82ae6971e1e8d4d55c84b3ca0a09fe",
        "benchmark_sha256": "f84052900331a508fca5a2332a9537d99ea03d2aa97cb537b85132324844aac2",
        "run_config_sha256": "09b3395fd9ca0e0a351e33c9327d0bed2a96ab70f04613fefbb57503cbc4438c",
        "quality_sha256": "9fbdc7f38dcf9ae999e2b3d3446a6e5cf43b83bf03685651ce277b01f6cc3365",
        "collection_sha256": "e2c3d45b957b6261c0cad322c82327fb91972bd80cfa095cc1b328c7ab3b1b71",
        "assess_sha256": "f73fd60644911fe254680b79e1d373d38e75e0fc14a41fbd54512e241fb8930b",
        "total_s": 251.95,
    },
    "R03": {
        "name": "20260914-045221-cache_ca_r03_teacache_t0016",
        "launch_sha256": "7ecb20676ff681fdfbbceab3a650fa900aa075e5eddabf35e6ae4458d0e7c002",
        "manifest_sha256": "1fdeccc38e5702e15dd37cd52f933ccd09d509ccbb3baa12705ee1dc04577cc4",
        "metadata_sha256": "47bd63e3bb6373e05f7838daff102620839b0e348e95febbe9da4d99dc8a26d9",
        "benchmark_sha256": "f5e3e7154f492d9d0e57f86a17d1e9d3faa6bbf6a840b5eb854b58df86cbaa49",
        "run_config_sha256": "d6c90681de8062cf937b468d6db6d6f9ee221a77f2b4b6914a185ba31cad8e90",
        "quality_sha256": "5699cc1a5ba43df4cfd92cf02b57ec5641623c89a6c0b1ad30980ef032651a7b",
        "collection_sha256": "6970acd8d69d779ea38fc7dd95277848018a868c676cee9fa04ff8ce279201f6",
        "assess_sha256": "c80f89563afc5057409de92e6c9c6a46be442ef72e658dfe3ad949442ad2f193",
        "total_s": 370.52,
    },
    "R04": {
        "name": "20260914-051802-cache_ca_r04_taylorseer_t0040_o1",
        "launch_sha256": "4795bcb36323aadcab8b54b176ab2a969441ccef470fd80181e02e5360349e0b",
        "manifest_sha256": "94091ab4de006122f1616c7cf93872c6bbf9dd65fb40f45100c81fdd62380246",
        "metadata_sha256": "998c906d5a6e26ce09ed6b06ce99f705553d3e60f92c1697759330f019d648d6",
        "benchmark_sha256": "5a2eccb0b9aa49c657689793b545642ba783a1b798019c2c6ac3ebc40b613cc4",
        "run_config_sha256": "1a3f75560f3c178358e3cb1859b893120e66b510907fd9275400745e5149dfd1",
        "quality_sha256": "71580feef142539a8e2fbecb7726518a2574b09531dcf337b536f860edd5f17e",
        "collection_sha256": "599d475eb6b801b1f1f1eac54041a7ecd5edf7ac410855db60d20e28f9bb8614",
        "assess_sha256": "4a9cebe56988e64358428d0ef777d740eb8ece9fdd30f3a02fc21a31d446c94d",
        "total_s": 382.27,
    },
    "R05": {
        "name": "20260914-054324-cache_ca_r05_taylorseer_t0180_w10_o1",
        "launch_sha256": "392f725844f3606360c56624126f169e98c5a376f0c5e17cd36f911d50ca487e",
        "manifest_sha256": "3d1cddb42c2f04fb2c387c81fa00d9dee14b1ff6c3432b74c5ba37f61b585342",
        "metadata_sha256": "598904fecc09e1d4d264f6787e99a8b8ccc778c1679792af289684ff8a5e387e",
        "benchmark_sha256": "8476c72990b29e4a179a4a925894404d24864caed49b178ca22ccc8e33478411",
        "run_config_sha256": "0c98e68c6aa33fbb2a6cc1f3d0041108206439efb915af97557aa92ebcfded51",
        "quality_sha256": "40019c7439645fcf97541fd5938476e865601189800dc28b377a73aa526273b8",
        "collection_sha256": "a01a58ef32d7eff8fe9412fca71b5ba6d8dae2c0acd7528967c6680ed391a5ce",
        "assess_sha256": "b2ec773ba16490ef6552a85465fd6c6bb6d777cf9b7b776a5a110251406db868",
        "total_s": 371.36,
    },
    "R06": {
        "name": "20260914-063907-cache_ca_r06_easycache_t010_w30",
        "launch_sha256": "e214d22a26dab33c3c1e9de79e9089ad5e40bb27a4b06203052527af43c702b7",
        "manifest_sha256": "fce503fabbb219c9cab0df441b6555163eb3e66c115e9f7119febd46d9530b9f",
        "metadata_sha256": "268d587870279c18bdf7a71682122315eb1370b12ea3d0d593368a00cd8ceae3",
        "benchmark_sha256": "f31a3cf05715bda16b4576931e33b95b43ec07fcbcea46bb2fff39efb9b0f206",
        "run_config_sha256": "6f32cda5b35e547e6d66e39b1b46e892526940aaa5f1619fa4b44d4de7e02880",
        "quality_sha256": "0627682df8bcefc782006e52ea6d864cdccd40a0c276f1ebe55747726defc570",
        "collection_sha256": "0ae49de1f193b6400093542cd25a5da49ad0e9f43e949e4d4b9a6b0533bacc88",
        "assess_sha256": "15ad43895f67e2693ea2b327ced9f39b232d2eff386b07729944ad83620f12b1",
        "total_s": 480.37,
    },
    "R07": {
        "name": "20260914-072228-cache_ca_r07_teacache_t00087",
        "launch_sha256": "3feae5a534052e7dd2ba4bc5387a41bdc207e0706081fecfdb73345339c46548",
        "manifest_sha256": "32edc63c661b28f0dd6c63b09a47d3e388a272a4451f2cf6f54ef2943f345a09",
        "metadata_sha256": "0ab88d8a3f77ff15d39ae6c6bf56076b88a1c3b60d2a8dd788376c6c63fe1d29",
        "benchmark_sha256": "ee4d80b76c56cfa9a2eab2b5921949204411a5dc7e5ab149f9b92a09514524e4",
        "run_config_sha256": "cdf42836a2de2044f42409fbe94fe9ea1990e73eeb6050d67a8dbad19a93a3e8",
        "quality_sha256": "3125086633bd8454bcc4072b921201fd7ab47537009ec85144b483154637315c",
        "collection_sha256": "223cc39506bf46bc9194bdbc5fb1b819caf0971a7966bbc571fc575b2921230a",
        "assess_sha256": "efdcb228ccb39432428292031e307dc69bb487a8dbf5a1c90997697a6f4edf91",
        "total_s": 480.18,
    },
    "R08": {
        "name": "20260914-083326-cache_ca_r08_taylorseer_t0180_w30_o1",
        "launch_sha256": "6c8f610c4e1f6a0085cde0bb56f7b80f0133f57725f8d6375d2f54644823271f",
        "manifest_sha256": "7b77a2d868e1257da3635901b46c8ae552c762b3e10532d555e700ca39c1ece9",
        "metadata_sha256": "b9689f9e763827a732e9d014e57d5b9f23cd3ef25f0ad60a6c9438b36c706da5",
        "benchmark_sha256": "a792aa773bbf2e921c0d160d183d9f09ecf2e1b8fcbc0b6b58e8be89b20c2f8d",
        "run_config_sha256": "2eac08b00536f71a6f0d5359f91114d198f7cd71c496bccad18a6d37c54d0989",
        "quality_sha256": "d6dc2272da2db234d1d2c28945ef5409eb7458858847c1697957c04cf4188bcb",
        "collection_sha256": "89129455be7990f6855e01bb14d5311ac3e429c9d59420b70e2d376cc1661862",
        "assess_sha256": None,
        "total_s": 480.96,
    },
}
# R01 can only run again after its first post hashes and receipt are explicitly
# pinned as a trusted predecessor. Its old prehash pins deliberately fail now.
# R02-R07 were separately reviewed before activation. R08 is only staged here;
# the running broker and remote trusted worker remain on the prior hash until
# separate coordinator review/sync. Every repeat needs a trusted post chain.
ENABLED_RUN_KEYS = frozenset(("R01", "R02", "R03", "R04", "R05", "R06", "R07", "R08"))
SNAPSHOT_FILES = (
    "metadata.json", "outputs/benchmark.json", "outputs/quality.json",
    "outputs/collection.json", "assess_verdict.json",
)
SYNC_FILES = ("outputs/benchmark.json", "outputs/quality.json",
              "outputs/collection.json", "assess_verdict.json")


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def key_for_run(run: Path) -> str:
    resolved = run.resolve()
    if (run.is_symlink() or resolved.parent != CACHE / "runs"
            or not resolved.is_dir()):
        raise ValueError("quality route requires the real seq0002 Cache run parent")
    for key, spec in EXPECTED_BUNDLES.items():
        if resolved.name == spec["name"]:
            return key
    raise ValueError("quality route run ID is outside the exact R01/R02/R03/R04/R05/R06/R07/R08 allowlist")


def validate_local(run: Path) -> tuple[str, dict]:
    key = key_for_run(run)
    if key not in ENABLED_RUN_KEYS:
        raise ValueError("quality route requires a coordinator-reviewed exact run")
    spec = EXPECTED_BUNDLES[key]
    exact = {
        "launch.sh": spec["launch_sha256"],
        "manifest.resolved.toml": spec["manifest_sha256"],
        "metadata.json": spec["metadata_sha256"],
        "outputs/benchmark.json": spec["benchmark_sha256"],
        "outputs/run_config.json": spec["run_config_sha256"],
        "outputs/quality.json": spec["quality_sha256"],
        "outputs/collection.json": spec["collection_sha256"],
        "assess_verdict.json": spec["assess_sha256"],
    }
    for relative, expected in exact.items():
        path = run / relative
        if expected is None:
            if path.exists() or path.is_symlink():
                raise ValueError(f"quality route expected absent predecessor: {relative}")
            continue
        if not path.is_file() or path.is_symlink() or file_sha(path) != expected:
            raise ValueError(f"quality route real-run SHA mismatch: {relative}")
    return key, spec


def snapshot_local(run: Path, key: str) -> tuple[Path, dict]:
    marker = utc().replace(":", "").replace("-", "").replace("+00:00", "Z") + f"_{os.getpid()}"
    target = PROJECT / "artifacts/phase3/pre_quality_snapshots/host" / key / marker
    target.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for relative in SNAPSHOT_FILES:
        source = run / relative
        if not source.is_file():
            continue
        if source.is_symlink():
            raise RuntimeError(f"host quality snapshot source is a symlink: {relative}")
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        observed = file_sha(destination)
        if observed != file_sha(source):
            raise RuntimeError(f"host pre-quality snapshot differs: {relative}")
        hashes[relative] = observed
    (target / "manifest.json").write_text(json.dumps({"run_dir": str(run), "run_key": key,
                                                      "captured_at_utc": utc(), "sha256": hashes},
                                                     indent=2, sort_keys=True) + "\n")
    return target, hashes


def trusted_remote_worker() -> dict:
    if (not LOCAL_WORKER.is_file() or LOCAL_WORKER.is_symlink()
            or file_sha(LOCAL_WORKER) != TRUSTED_WORKER_SHA256):
        raise RuntimeError("host trusted quality worker SHA mismatch")
    remote_worker_sha = site_remote.ssh("sha256sum", str(REMOTE_WORKER), timeout=20).stdout.split()[0]
    remote_python_sha = site_remote.ssh("sha256sum", REMOTE_PYTHON, timeout=20).stdout.split()[0]
    if remote_worker_sha != TRUSTED_WORKER_SHA256 or remote_python_sha != TRUSTED_PYTHON_SHA256:
        raise RuntimeError("remote trusted quality worker/interpreter SHA mismatch")
    return {"worker_sha256": remote_worker_sha, "interpreter_sha256": remote_python_sha}


def remote_run(run: Path, key: str) -> tuple[dict, int]:
    try:
        result = site_remote.ssh(REMOTE_PYTHON, str(REMOTE_WORKER), key, timeout=1800)
        raw, code = result.stdout, result.returncode
    except subprocess.CalledProcessError as exc:
        raw, code = exc.stdout or "", exc.returncode
    summary = site_remote._json_receipt(raw)
    return summary, code


def pull_attempt(run: Path, key: str, summary: dict) -> Path | None:
    raw = summary.get("attempt_dir")
    if raw is None:
        return None
    remote = Path(raw)
    expected_parent = REMOTE_CACHE / "runs" / EXPECTED_BUNDLES[key]["name"] / "phase3_quality_attempts"
    if (remote.parent != expected_parent or not re.fullmatch(
            r"[0-9]{8}T[0-9]{6}\.[0-9]{6}\+0000_[0-9]+", remote.name)):
        raise RuntimeError("remote quality attempt path is outside the exact run")
    local = run / "phase3_quality_attempts" / remote.name
    if local.exists() or local.is_symlink():
        raise RuntimeError("quality attempt local destination already exists")
    site_remote.pull_tree(remote, local)
    receipt = local / "receipt.json"
    if not receipt.is_file() or file_sha(receipt) != summary.get("receipt_sha256"):
        raise RuntimeError("pulled quality receipt SHA mismatch")
    return local


def check_post_benchmark(path: Path, expected_total: float) -> None:
    data = json.loads(path.read_text())
    prompts = data.get("per_prompt")
    if (not isinstance(prompts, list) or len(prompts) != 5
            or data.get("timing_scope") !=
            "five_prompt_sum_sglang_request_metrics_excluding_model_load_internal_one_step_warmup_and_video_export"
            or (data.get("export") or {}).get("frame_count") != 80):
        raise RuntimeError("post-quality benchmark lost frozen five-prompt fields")
    total = float(data["total_s"])
    summed = sum(float(row["request_s"]) for row in prompts)
    if abs(total - expected_total) > 1e-3 or abs(total - summed) > 1e-5:
        raise RuntimeError("post-quality timing numerator/denominator changed")


def sync_small_post(run: Path, attempt: Path, pre_hashes: dict, expected_total: float) -> dict:
    source_root = attempt / "post"
    for relative in SYNC_FILES:
        source = source_root / relative
        target = run / relative
        if not source.is_file() or source.is_symlink():
            raise RuntimeError(f"small quality post file missing: {relative}")
        if (target.is_symlink() or (relative in pre_hashes and
                                   (not target.is_file() or file_sha(target) != pre_hashes[relative]))
                or (relative not in pre_hashes and target.exists())):
            raise RuntimeError(f"local quality file changed during remote evaluation: {relative}")
    check_post_benchmark(source_root / "outputs/benchmark.json", expected_total)
    copied = {}
    for relative in SYNC_FILES:
        source = source_root / relative
        target = run / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if file_sha(target) != file_sha(source):
            raise RuntimeError(f"quality JSON sync SHA mismatch: {relative}")
        copied[relative] = file_sha(target)
    return copied


def assess(run_dir: Path) -> dict:
    key, spec = validate_local(run_dir)
    run = run_dir.resolve()
    source = trusted_remote_worker()
    pre_dir, pre_hashes = snapshot_local(run, key)
    started_utc, start_ns = utc(), time.monotonic_ns()
    receipt: dict[str, object] = {"status": "transport_failed", "run_key": key,
                                  "run_dir": str(run), "started_at_utc": started_utc,
                                  "trusted_source": source, "host_pre_snapshot_dir": str(pre_dir),
                                  "host_pre_sha256": pre_hashes,
                                  "gpu_authorization": "not_granted_by_quality_route"}
    try:
        summary, exit_code = remote_run(run, key)
        receipt["remote_exit_code"] = exit_code
        receipt["remote_summary"] = summary
        attempt = pull_attempt(run, key, summary)
        receipt["local_attempt_dir"] = str(attempt) if attempt else None
        if attempt is not None:
            remote_receipt = json.loads((attempt / "receipt.json").read_text())
            receipt["remote_pre_sha256"] = remote_receipt.get("pre_snapshot")
            receipt["remote_post_sha256"] = remote_receipt.get("post_snapshot")
        if exit_code == 0 and summary.get("status") == SUCCESS and attempt is not None:
            receipt["host_post_sha256"] = sync_small_post(run, attempt, pre_hashes, spec["total_s"])
            receipt["status"] = SUCCESS
        else:
            receipt["status"] = "remote_quality_failed"
    except Exception as exc:
        receipt["status"] = "transport_failed"
        receipt["error_type"] = type(exc).__name__
        receipt["error"] = str(exc)
    receipt["finished_at_utc"] = utc()
    receipt["host_wall_s"] = (time.monotonic_ns() - start_ns) / 1e9
    host_receipt = pre_dir / "transport_receipt.json"
    host_receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    receipt["host_receipt"] = str(host_receipt)
    receipt["host_receipt_sha256"] = file_sha(host_receipt)
    if receipt["status"] != SUCCESS:
        raise RuntimeError(f"typed CPU quality failed; see {host_receipt}")
    return receipt
