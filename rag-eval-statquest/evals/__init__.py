# Windows consoles default to cp1252, which can't encode the emoji DeepEval
# prints in its results (e.g. U+2728), crashing evaluate() after all the LLM
# calls are done. Force UTF-8 on stdout/stderr for every eval run.
import sys

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
