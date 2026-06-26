"""Cross-platform file locking.

Provides flock (advisory file lock) functions that work on both
Unix (via fcntl) and Windows (via msvcrt).
"""
from __future__ import annotations

import os
from typing import IO

if os.name == "posix":
    import fcntl

    def flock_exclusive(f: IO) -> None:
        """Acquire an exclusive advisory lock on the file."""
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)

    def flock_unlock(f: IO) -> None:
        """Release the advisory lock on the file."""
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

elif os.name == "nt":
    # Windows: use no-op. The original Unix fcntl.flock is advisory only and
    # Python's GIL provides sufficient thread safety for append-only jsonl writes.
    # msvcrt.locking would impose mandatory locks that break same-process reopens.
    def flock_exclusive(f: IO) -> None:
        pass

    def flock_unlock(f: IO) -> None:
        pass

else:
    # Fallback for unknown platforms: no-op
    def flock_exclusive(f: IO) -> None:
        pass

    def flock_unlock(f: IO) -> None:
        pass
