import sys
import traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))
from robust_cvar_portfolio.data.loader import load_dataset, build_state_matrix
ROOT = Path(__file__).resolve().parents[1]
try:
    b = load_dataset(ROOT / "configs" / "sp30.yaml", ROOT / "data" / "processed")
    print("returns", b["returns"].shape)
    s = build_state_matrix(b["returns"])
    print("states", s.shape)
except Exception:
    traceback.print_exc()
