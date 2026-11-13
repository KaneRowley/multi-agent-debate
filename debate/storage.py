"""Save complete JSON records without replacing existing records."""
import errno
import hashlib
import json
import os
import tempfile
from pathlib import Path


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False,
            separators=(",", ":")
        ).encode()
    ).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=".pending-"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(
                value, f, indent=2,
                ensure_ascii=False, allow_nan=False
            )
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())

        try:
            os.link(temporary, path)
        except OSError as exc:
            on_drive = path.resolve().is_relative_to(
                Path("/content/drive")
            )
            unsupported = exc.errno in (
                errno.EPERM, errno.EOPNOTSUPP, errno.EXDEV
            )
            if not (on_drive and unsupported):
                raise
            # The runner holds the exclusive run-directory lock.
            if path.exists():
                raise FileExistsError(str(path))
            os.rename(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
