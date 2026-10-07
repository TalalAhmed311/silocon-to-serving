import sys
from pathlib import Path

V1 = Path(__file__).resolve().parents[1]
if str(V1) not in sys.path:
    sys.path.insert(0, str(V1))
