import os
import tempfile
from pathlib import Path

# Point traces at a temp dir, and the knowledge base at the bundled sample docs,
# before anything imports settings.
_tmp = Path(tempfile.mkdtemp(prefix="introspective-tests-"))
os.environ["TRACE_DIR"] = str(_tmp / "traces")
os.environ.pop("DOCS_DIR", None)
