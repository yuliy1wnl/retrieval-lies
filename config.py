# ── Endee connection ──────────────────────────────────────────────────────────
ENDEE_BASE_URL = "http://localhost:8080/api/v1"
ENDEE_API_TOKEN = ""

# ── Dataset ───────────────────────────────────────────────────────────────────
NUM_PASSAGES = 25_000
NUM_QUERIES   = 200
DATA_DIR      = "data/msmarco"

# ── Embedding models to benchmark ─────────────────────────────────────────────
EMBEDDING_MODELS = [
    "all-MiniLM-L6-v2",        # fast baseline,  384-dim
    "BAAI/bge-small-en-v1.5",  # stronger,       384-dim
    "intfloat/e5-small-v2",    # different objective, 384-dim
]
EMBEDDING_DIM = 384

# ── Endee index configurations to benchmark ───────────────────────────────────
INDEX_CONFIGS = [
    {"precision": "float32", "ef_con": 128, "M": 16},
    {"precision": "float32", "ef_con": 64,  "M": 16},
    {"precision": "int16",   "ef_con": 128, "M": 16},
    {"precision": "int16",   "ef_con": 64,  "M": 16},
    {"precision": "int8",    "ef_con": 128, "M": 16},
    {"precision": "int8",    "ef_con": 64,  "M": 16},
]

# ── Evaluation settings ───────────────────────────────────────────────────────
TOP_K_VALUES  = [1, 3, 5, 10]
EF_SEARCH     = 128
BATCH_SIZE    = 256   # embedding batch size — fits comfortably in 8GB VRAM

# ── Output ────────────────────────────────────────────────────────────────────
RESULTS_DIR = "results"
REPORTS_DIR = "reports"