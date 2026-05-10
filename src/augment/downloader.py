"""Download + extract augmentation corpora.

- MIT IR Survey: room impulse responses (mirror on HF for reproducibility)
- MUSAN: noise, music, speech (~11 GB)
- FSD50K: environmental sounds (~34 GB compressed)
- Common Voice 17.0: streamed from HF, samples written as wavs

All callers should treat downloads as resumable + idempotent. Each dataset
records a `.complete` sentinel so we don't re-extract.
"""
from __future__ import annotations

import logging
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Callable, Iterable

import requests
from huggingface_hub import snapshot_download
from tqdm import tqdm

from src.settings import get_settings

logger = logging.getLogger(__name__)


ProgressCallback = Callable[[str, float], None]
"""(dataset_name, fraction_done) -> None"""


# -------- low-level helpers --------

def _download_with_progress(
    url: str,
    dest: Path,
    progress: ProgressCallback | None = None,
    name: str = "",
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        bytes_done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                f.write(chunk)
                bytes_done += len(chunk)
                if progress and total:
                    progress(name, bytes_done / total)
    tmp.rename(dest)


def _extract_archive(archive: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    if archive.suffixes[-2:] == [".tar", ".gz"] or archive.suffix == ".tgz":
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(target_dir)
    elif archive.suffix == ".zip":
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(target_dir)
    elif archive.suffix == ".tar":
        with tarfile.open(archive, "r:") as tf:
            tf.extractall(target_dir)
    else:
        raise ValueError(f"Unknown archive type: {archive}")


def _mark_complete(target_dir: Path) -> None:
    (target_dir / ".complete").write_text("ok")


def _is_complete(target_dir: Path) -> bool:
    return (target_dir / ".complete").exists()


# -------- MIT IR Survey --------

MIT_RIRS_HF_REPO = "davidscripka/MIT_environmental_impulse_responses"


def download_mit_rirs(progress: ProgressCallback | None = None) -> Path:
    """Download MIT Reverb / IR Survey via davidscripka's HF mirror."""
    target = get_settings().rirs_dir / "mit"
    if _is_complete(target):
        return target
    target.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading MIT IR Survey to %s", target)
    if progress:
        progress("mit_rirs", 0.05)
    snapshot_download(
        repo_id=MIT_RIRS_HF_REPO,
        repo_type="dataset",
        local_dir=str(target),
        local_dir_use_symlinks=False,
    )
    if progress:
        progress("mit_rirs", 1.0)
    _mark_complete(target)
    return target


# -------- MUSAN --------

MUSAN_URL = "https://us.openslr.org/resources/17/musan.tar.gz"


def download_musan(progress: ProgressCallback | None = None) -> Path:
    """Download + extract MUSAN. ~11 GB tarball."""
    target = get_settings().musan_dir
    if _is_complete(target):
        return target
    target.mkdir(parents=True, exist_ok=True)
    archive = target / "musan.tar.gz"
    logger.info("Downloading MUSAN to %s", archive)
    _download_with_progress(MUSAN_URL, archive, progress, name="musan")
    logger.info("Extracting MUSAN")
    if progress:
        progress("musan", 0.95)
    _extract_archive(archive, target)
    archive.unlink(missing_ok=True)
    _mark_complete(target)
    if progress:
        progress("musan", 1.0)
    return target


# -------- FSD50K --------

FSD50K_BASE = "https://zenodo.org/records/4060432/files"
FSD50K_PARTS = [
    "FSD50K.dev_audio.zip",
    "FSD50K.eval_audio.zip",
    "FSD50K.ground_truth.zip",
]


def download_fsd50k(
    progress: ProgressCallback | None = None,
    parts: Iterable[str] = FSD50K_PARTS,
) -> Path:
    """Download FSD50K dev + eval audio + labels.

    NOTE: dev_audio is a multi-part zip on Zenodo (z01..z05 + .zip). Zenodo also
    serves a single `FSD50K.dev_audio.zip` reconstruction; we use that.
    """
    target = get_settings().fsd50k_dir
    if _is_complete(target):
        return target
    target.mkdir(parents=True, exist_ok=True)

    parts = list(parts)
    n = len(parts)
    for i, name in enumerate(parts):
        url = f"{FSD50K_BASE}/{name}"
        archive = target / name
        logger.info("Downloading %s", url)
        _download_with_progress(url, archive, progress, name=f"fsd50k:{name}")
        logger.info("Extracting %s", name)
        _extract_archive(archive, target)
        archive.unlink(missing_ok=True)
        if progress:
            progress("fsd50k", (i + 1) / n)
    _mark_complete(target)
    return target


# -------- Common Voice 17 (streamed) --------

COMMON_VOICE_REPO = "mozilla-foundation/common_voice_17_0"


def download_common_voice_subset(
    n_samples: int,
    language: str = "en",
    progress: ProgressCallback | None = None,
) -> Path:
    """Stream a subset of Common Voice clips and write them as 16 kHz wavs.

    Streaming avoids downloading the full ~80 GB English split. Caller needs an
    HF token with the Common Voice ToS accepted (settings.hf_token).
    """
    import numpy as np
    import soundfile as sf
    from datasets import load_dataset

    target = get_settings().common_voice_dir / language
    if _is_complete(target):
        return target
    target.mkdir(parents=True, exist_ok=True)

    settings = get_settings()
    token = settings.hf_token

    logger.info("Streaming %d Common Voice clips (%s)", n_samples, language)
    ds = load_dataset(
        COMMON_VOICE_REPO,
        language,
        split="train",
        streaming=True,
        token=token,
        trust_remote_code=True,
    )

    written = 0
    pbar = tqdm(total=n_samples, desc=f"common_voice:{language}")
    for row in ds:
        if written >= n_samples:
            break
        audio = row.get("audio")
        if not audio or audio.get("array") is None:
            continue
        arr = np.asarray(audio["array"], dtype=np.float32)
        sr = int(audio.get("sampling_rate", 16_000))
        if sr != 16_000:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(sr, 16_000)
            arr = resample_poly(arr, 16_000 // g, sr // g).astype(np.float32)
        out_path = target / f"cv_{written:07d}.wav"
        sf.write(out_path, arr, 16_000, subtype="PCM_16")
        written += 1
        pbar.update(1)
        if progress and written % 50 == 0:
            progress("common_voice", written / max(1, n_samples))
    pbar.close()
    _mark_complete(target)
    if progress:
        progress("common_voice", 1.0)
    return target


# -------- top-level entry --------

def ensure_corpora(
    *,
    use_mit_rirs: bool,
    use_musan: bool,
    use_fsd50k: bool,
    use_common_voice: bool,
    common_voice_subset: int = 10_000,
    progress: ProgressCallback | None = None,
) -> dict[str, Path]:
    """Download whichever corpora the run config asks for."""
    out: dict[str, Path] = {}
    if use_mit_rirs:
        out["mit_rirs"] = download_mit_rirs(progress)
    if use_musan:
        out["musan"] = download_musan(progress)
    if use_fsd50k:
        out["fsd50k"] = download_fsd50k(progress)
    if use_common_voice:
        out["common_voice"] = download_common_voice_subset(common_voice_subset, progress=progress)
    return out
