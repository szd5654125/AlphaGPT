import os
import torch


class DatabaseConfig:
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "crypto_quant")
    DB_DSN = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    CHAIN = "solana"
    TIMEFRAME = "1m" # 也支持 15min
    MIN_LIQUIDITY_USD = 500000.0
    MIN_FDV = 10000000.0
    MAX_FDV = float('inf')
    BIRDEYE_API_KEY = os.getenv("BIRDEYE_API_KEY", "")
    BIRDEYE_IS_PAID = True
    USE_DEXSCREENER = False
    CONCURRENCY = 20
    HISTORY_DAYS = 7


class ModelConfig:
    DB_URL = f"postgresql://{os.getenv('DB_USER','postgres')}:{os.getenv('DB_PASSWORD','password')}@{os.getenv('DB_HOST','localhost')}:5432/{os.getenv('DB_NAME','crypto_quant')}"
    BATCH_SIZE = 2048
    TRAIN_STEPS = 1000
    MAX_FORMULA_LEN = 12
    TRADE_SIZE_USD = 1000.0
    MIN_LIQUIDITY = 5000.0 # 低于此流动性视为归零/无法交易
    BASE_FEE = 0.005 # 基础费率 0.5% (Swap + Gas + Jito Tip)
    THRESH_BINS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
    # variable-length formula
    MIN_FORMULA_LEN = 4  # 至少生成多少步才允许 stop
    OPS_PENALTY_LAMBDA = 0.0001  # reward -= λ * (#ops)
    STOP_PROB_EPS = 1e-6  # clamp stop prob for numerical stability
# --- Actor-Critic (baseline) ---
    VALUE_LOSS_COEF = 0.01      # critic loss 权重：先小一点更稳
    ADV_NORM_EPS = 1e-5        # advantage 标准化用
    USE_LORD = True
    LORD_DECAY_RATE = 1e-3
    LORD_NUM_ITERATIONS = 5
    OUTPUT_DIR = "."  # 例如 "runs/exp1"
    # Multi-GPU run (optional)
    RUN_MULTI_GPU = False
    GPUS = [0, 1, 2]  # 物理 GPU id（给 CUDA_VISIBLE_DEVICES 用）
    SEEDS = [100, 101, 102, 103, 104, 105]  # 长度需等于 len(GPUS) * SEEDS_PER_GPU
    SEEDS_PER_GPU = 2
    CHILD_DEVICE = "cuda:0"  # 子进程里固定用 cuda:0（由 CUDA_VISIBLE_DEVICES 映射）


class StrategyConfig:
    MAX_OPEN_POSITIONS = 3
    ENTRY_AMOUNT_SOL = 2.0
    STOP_LOSS_PCT = -0.05
    TAKE_PROFIT_Target1 = 0.10
    TP_Target1_Ratio = 0.5
    TRAILING_ACTIVATION = 0.05
    TRAILING_DROP = 0.03
    BUY_THRESHOLD = 0.85
    SELL_THRESHOLD = 0.45


FEATURE_PM1_SPECS = {
    # --- core factors ---
    "ret":         {"kind": "robust_z_softsign"},
    "liq_score":   {"kind": "bounded_01"},
    "pressure":    {"kind": "identity_pm1"},
    "fomo":        {"kind": "robust_z_softsign"},
    "dev":         {"kind": "robust_z_softsign"},
    "log_vol":     {"kind": "robust_z_softsign"},
    # --- advanced ---
    "vol_cluster": {"kind": "robust_z_softsign"},
    "mom_sign":    {"kind": "binary_sign"},   # 你在 factors.py 里生成 {-1,+1}
    "rel_strength":{"kind": "bounded_0_100"},
    "momentum_rev": {"kind": "bounded_01"},
    "hl_range":    {"kind": "robust_z_softsign"},
    "close_pos":   {"kind": "bounded_01"},
    "vol_trend":   {"kind": "robust_z_softsign"}}


class ExecutionConfig:
    '''RPC_URL = os.getenv("QUICKNODE_RPC_URL", "填入RPC地址")
    _PRIV_KEY_STR = os.getenv("SOLANA_PRIVATE_KEY", "")

    if not _PRIV_KEY_STR:
        raise ValueError("Missing SOLANA_PRIVATE_KEY in .env")
    try:
        PAYER_KEYPAIR = Keypair.from_base58_string(_PRIV_KEY_STR)
    except Exception:
        import json
        PAYER_KEYPAIR = Keypair.from_bytes(json.loads(_PRIV_KEY_STR))
    WALLET_ADDRESS = str(PAYER_KEYPAIR.pubkey())'''
    DEFAULT_SLIPPAGE_BPS = 200  # bps
    PRIORITY_LEVEL = "High"
    SOL_MINT = "So11111111111111111111111111111111111111112"
    USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"