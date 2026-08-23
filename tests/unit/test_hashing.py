from pathlib import Path

from ranklab.artifacts.hashing import sha256_file


def test_sha256_file_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "artifact.txt"
    path.write_bytes(b"ranklab\n")
    assert sha256_file(path) == "ba660acfeec0c51144f8a0c3613d1835f930405ca34e17c0fc7d6d679c50f253"
