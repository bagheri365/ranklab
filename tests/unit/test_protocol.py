from pathlib import Path

import pytest

from ranklab.benchmark.protocol import ProtocolNotFrozenError, require_frozen_protocol


def test_unfrozen_protocol_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "protocol.yaml"
    path.write_text("status: UNFROZEN\n")
    with pytest.raises(ProtocolNotFrozenError):
        require_frozen_protocol(path)
