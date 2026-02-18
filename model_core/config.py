import os
import torch

class ModelConfig:
    # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DEVICE = torch.device("cuda:2")
    DB_URL = f"postgresql://{os.getenv('DB_USER','postgres')}:{os.getenv('DB_PASSWORD','password')}@{os.getenv('DB_HOST','localhost')}:5432/{os.getenv('DB_NAME','crypto_quant')}"
    BATCH_SIZE = 2048
    TRAIN_STEPS = 1000
    MAX_FORMULA_LEN = 12
    TRADE_SIZE_USD = 1000.0
    MIN_LIQUIDITY = 5000.0 # 低于此流动性视为归零/无法交易
    BASE_FEE = 0.005 # 基础费率 0.5% (Swap + Gas + Jito Tip)
    INPUT_DIM = 6
    THRESH_BINS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

"""
每个输入特征的统一缩放/映射规则（目标域：[-1, 1]）。

kind 约定：
  - robust_z_softsign:  median/MAD -> clamp(z) -> tanh(z/scale)
  - bounded_01:     [0,1] -> [-1,1]
  - identity_pm1:   已经在 [-1,1]，只做 clamp
  - binary_sign:    输出必须是 {-1,+1}（在 factors.py 中生成）
"""

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