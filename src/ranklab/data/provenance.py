from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from ranklab.artifacts.hashing import md5_file, sha256_file

KUAIRAND_PURE_RECORD_URL = "https://zenodo.org/records/10439422"
KUAIRAND_PURE_ARCHIVE_URL = (
    "https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz"
)
KUAIRAND_PURE_ARCHIVE_NAME = "KuaiRand-Pure.tar.gz"
KUAIRAND_PURE_ARCHIVE_MD5 = "0820331067a3784d9691136f772b35a7"
KUAIRAND_PURE_VERSION = "v1"

KUAIRAND_PURE_EXPECTED_FILES: tuple[str, ...] = (
    "log_random_4_22_to_5_08_pure.csv",
    "log_standard_4_08_to_4_21_pure.csv",
    "log_standard_4_22_to_5_08_pure.csv",
    "user_features_pure.csv",
    "video_features_basic_pure.csv",
    "video_features_statistic_pure.csv",
)


@dataclass(frozen=True)
class FileProvenance:
    name: str
    bytes: int
    sha256: str


def verify_archive(path: str | Path) -> bool:
    """Verify the official KuaiRand-Pure archive against the published MD5."""
    return md5_file(path) == KUAIRAND_PURE_ARCHIVE_MD5


def inventory_data_dir(data_dir: str | Path) -> list[FileProvenance]:
    """Return deterministic provenance records for the six expected Pure CSV files."""
    root = Path(data_dir)
    missing = [name for name in KUAIRAND_PURE_EXPECTED_FILES if not (root / name).is_file()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"missing expected KuaiRand-Pure files: {joined}")

    return [
        FileProvenance(
            name=name,
            bytes=(root / name).stat().st_size,
            sha256=sha256_file(root / name),
        )
        for name in KUAIRAND_PURE_EXPECTED_FILES
    ]


def inventory_as_dicts(data_dir: str | Path) -> list[dict[str, str | int]]:
    return [asdict(row) for row in inventory_data_dir(data_dir)]
