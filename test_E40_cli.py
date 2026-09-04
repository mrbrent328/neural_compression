"""Executable validation for the E40 command-line interface."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile


def test_cli_file_roundtrip():
    data = (b"CLI integration: persistent structure\n" * 1000) + bytes(range(256))
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "source.bin"
        encoded = root / "source.nc40"
        restored = root / "restored.bin"
        source.write_bytes(data)

        encode_result = subprocess.run(
            [
                sys.executable, "E40_cli.py", "encode",
                str(source), str(encoded), "--region-size", "257",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "selected=" in encode_result.stdout

        subprocess.run(
            [
                sys.executable, "E40_cli.py", "decode",
                str(encoded), str(restored),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        assert restored.read_bytes() == data


if __name__ == "__main__":
    test_cli_file_roundtrip()
    print("PASS test_cli_file_roundtrip")
