import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))
from robust_cvar_portfolio.src.data_loader import download_prices
p = download_prices(["AAPL","MSFT","LIN","BRK.A"], "2010-01-01", "2024-12-31")
print(p.shape, p.index.min(), p.index.max())
