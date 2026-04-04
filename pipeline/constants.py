from pathlib import Path

FRONTEND_DIR = Path(__file__).parent.parent.resolve()   # frontEnd/
HISTORY_DIR  = FRONTEND_DIR / "history"

DEFAULT_SIZE = 512
DEFAULT_SNR  = 14
RTL_HARDCODED_SIZE = 512   # tb_wiener_top.sv is hardcoded for 512×512
