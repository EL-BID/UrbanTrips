from __future__ import annotations

import os
import zipfile
from contextlib import contextmanager
from typing import IO


@contextmanager
def open_csv(path: str) -> "contextmanager[IO]":
    """Open a CSV or CSV-in-ZIP for reading, skipping macOS metadata entries."""
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            members = [
                m
                for m in zf.namelist()
                if not m.endswith("/")
                and not m.startswith("__MACOSX")
                and not os.path.basename(m).startswith("._")
            ]
            if not members:
                raise FileNotFoundError(f"No CSV files found inside zip: {path}")
            csv_members = [m for m in members if m.lower().endswith(".csv")]
            member = csv_members[0] if csv_members else members[0]
            with zf.open(member) as f:
                yield f
    else:
        with open(path, "rb") as f:
            yield f


def resolve_zip(path: str) -> str:
    """Return path as-is if it exists, or path+'.zip' if the zip exists."""
    if not os.path.exists(path) and path.endswith(".csv"):
        zip_path = path + ".zip"
        if os.path.exists(zip_path):
            return zip_path
    return path
