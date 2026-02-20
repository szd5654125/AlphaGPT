import torch
import torch.nn as nn
from utils import shift1
from config.general_config import FEATURE_PM1_SPECS


_EPS = 1e-6


class RMSNormFactor(nn.Module):
    """RMSNorm for factor normalization"""
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class MemeIndicators:
    @staticmethod
    def liquidity_health(liquidity, fdv):
        ratio = liquidity / (fdv + 1e-6)
        return torch.clamp(ratio * 4.0, 0.0, 1.0)

    @staticmethod
    def buy_sell_imbalance(close, open_, high, low):
        range_hl = high - low + 1e-9
        body = close - open_
        strength = body / range_hl
        return (strength * 3.0) / (1.0 + torch.abs(strength * 3.0))

    @staticmethod
    def fomo_acceleration(volume, window=5):
        vol_prev = shift1(volume, fill=0.0)
        vol_chg = (volume - vol_prev) / (vol_prev + 1.0)
        acc = vol_chg - shift1(vol_chg, fill=0.0)
        return torch.clamp(acc, -5.0, 5.0)

    @staticmethod
    def pump_deviation(close, window=20):
        pad = close[:, :1].repeat(1, window-1)
        c_pad = torch.cat([pad, close], dim=1)
        ma = c_pad.unfold(1, window, 1).mean(dim=-1)
        dev = (close - ma) / (ma + 1e-9)
        return dev

    @staticmethod
    def volatility_clustering(close, window=10):
        """Detect volatility clustering patterns"""
        ret = torch.log(close / (shift1(close, fill=0.0) + 1e-9))
        ret_sq = ret ** 2
        
        pad = torch.zeros((ret_sq.shape[0], window-1), device=close.device)
        ret_sq_pad = torch.cat([pad, ret_sq], dim=1)
        vol_ma = ret_sq_pad.unfold(1, window, 1).mean(dim=-1)
        
        return torch.sqrt(vol_ma + 1e-9)

    @staticmethod
    def momentum_reversal(close, window=5):
        """Capture momentum reversal signals"""
        ret = torch.log(close / (shift1(close, fill=0.0) + 1e-9))
        
        pad = torch.zeros((ret.shape[0], window-1), device=close.device)
        ret_pad = torch.cat([pad, ret], dim=1)
        mom = ret_pad.unfold(1, window, 1).sum(dim=-1)
        
        # Detect reversals
        mom_prev = shift1(mom, fill=0.0)
        reversal = (mom * mom_prev < 0).float()
        
        return reversal

    @staticmethod
    def relative_strength(close, high, low, window=14):
        """RSI-like indicator for strength detection"""
        ret = close - shift1(close, fill=0.0)
        
        gains = torch.relu(ret)
        losses = torch.relu(-ret)
        
        pad = torch.zeros((gains.shape[0], window-1), device=close.device)
        gains_pad = torch.cat([pad, gains], dim=1)
        losses_pad = torch.cat([pad, losses], dim=1)
        
        avg_gain = gains_pad.unfold(1, window, 1).mean(dim=-1)
        avg_loss = losses_pad.unfold(1, window, 1).mean(dim=-1)
        
        rs = (avg_gain + 1e-9) / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi  # [0, 100]


def _pm1_from_z(z: torch.Tensor) -> torch.Tensor:
    """Map roughly-unbounded z to [-1,1] with tanh; scale controls saturation speed."""
    return z / (1.0 + torch.abs(z))


def _pm1_from_01(x: torch.Tensor) -> torch.Tensor:
    """Map [0,1] -> [-1,1]."""
    return torch.clamp(x * 2.0 - 1.0, -1.0, 1.0)


def _pm1_from_0_100(x: torch.Tensor) -> torch.Tensor:
    """Map [0,100] -> [-1,1] (e.g., RSI)."""
    return torch.clamp((x - 50.0) / 50.0, -1.0, 1.0)


def _pm1_from_neg100_100(x: torch.Tensor) -> torch.Tensor:
    """Map [-100,100] -> [-1,1] (e.g., some oscillators)."""
    return torch.clamp(x / 100.0, -1.0, 1.0)


def robust_norm(t, clamp= 5.0, eps= _EPS):
    """
    Median/MAD robust z-score, per-sample along time dim=1.
    Returns z in [-clamp, clamp].
    """
    median = torch.nanmedian(t, dim=1, keepdim=True)[0]
    mad = torch.nanmedian(torch.abs(t - median), dim=1, keepdim=True)[0] + eps
    z = (t - median) / mad
    return torch.clamp(z, -clamp, clamp)


def normalize_feature(name: str, x: torch.Tensor) -> torch.Tensor:
    """
    统一把特征映射到 [-1,1]。规则来自 FEATURE_PM1_SPECS。
    """
    spec = FEATURE_PM1_SPECS.get(name)
    if spec is None:
        raise KeyError(f"[feature_scaling] missing spec for: {name}")

    kind = spec["kind"]
    if kind == "robust_z_softsign":
        return _pm1_from_z(robust_norm(x))

    if kind == "bounded_01":
        return _pm1_from_01(x)

    if kind == "identity_pm1":
        return torch.clamp(x, -1.0, 1.0)

    if kind == "binary_sign":
        # 这里假设你传入的已经是 {-1,+1}
        return torch.clamp(x, -1.0, 1.0)

    if kind == "bounded_0_100":
        return _pm1_from_0_100(x)

    if kind == "bounded_neg100_100":
        return _pm1_from_neg100_100(x)

    raise ValueError(f"[feature_scaling] unknown kind={kind} for feature={name}")


class FeatureEngineer:
    INPUT_DIM = len(FEATURE_PM1_SPECS)

    @staticmethod
    def compute_features(raw_dict):
        c = raw_dict['close']
        o = raw_dict['open']
        h = raw_dict['high']
        l = raw_dict['low']
        v = raw_dict['volume']
        liq = raw_dict['liquidity']
        fdv = raw_dict['fdv']

          # 1) 先把“可能会用到”的 raw 特征都算出来
        raw = {}
        raw["ret"] = torch.log(c / (shift1(c, fill=0.0) + 1e-9))
        raw["liq_score"] = MemeIndicators.liquidity_health(liq, fdv)
        raw["pressure"] = MemeIndicators.buy_sell_imbalance(c, o, h, l)
        raw["fomo"] = MemeIndicators.fomo_acceleration(v)
        raw["dev"] = MemeIndicators.pump_deviation(c)
        raw["log_vol"] = torch.log1p(v)

        # advanced 可选（如果 FEATURE_KEYS 里启用了这些 key，就需要它们存在）
        raw["vol_cluster"] = MemeIndicators.volatility_clustering(c)
        raw["momentum_rev"] = MemeIndicators.momentum_reversal(c)
        raw["rel_strength"] = MemeIndicators.relative_strength(c, h, l)
        raw["hl_range"] = (h - l) / (c + 1e-9)
        raw["close_pos"] = (c - l) / (h - l + 1e-9)
        vol_prev = shift1(v, fill=0.0)
        raw["vol_trend"] = (v - vol_prev) / (vol_prev + 1.0)
        raw["mom_sign"] = torch.where(raw["ret"] >= 0, torch.ones_like(raw["ret"]), -torch.ones_like(raw["ret"]))
        # 2) 按配置选择通道并 normalize
        missing = [k for k in FEATURE_PM1_SPECS if k not in raw]
        if missing:
            raise KeyError(f"[FeatureEngineer] missing raw features for keys: {missing}")
        features = torch.stack([normalize_feature(k, raw[k]) for k in FEATURE_PM1_SPECS], dim = 1)
        return features