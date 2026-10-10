"""ai_press_hoops: basketball video analysis POC (CV perception -> System One decisions -> VLM reasoning)."""

import os

# Unsupported Apple-GPU ops fall back to CPU instead of crashing; must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

__version__ = "0.1.0"
