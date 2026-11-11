"""J.A.R.V.I.S. voice assistant: wake word, arc-reactor HUD and Mac tools, on top of OpenJarvis.

Run it with ``jarvis assistant``.
"""

import os
import sys


def main() -> None:
    # macOS keeps freed large allocations cached in the process, which roughly triples
    # Whisper's real memory use and stops it ever being given back. The allocator reads
    # this setting only at process start, so relaunch once with it set.
    if sys.platform == "darwin" and "MallocLargeCache" not in os.environ:
        os.environ["MallocLargeCache"] = "0"
        os.execv(sys.executable, [sys.executable, "-m", "openjarvis.assistant", *sys.argv[1:]])

    from .server import run

    run()
