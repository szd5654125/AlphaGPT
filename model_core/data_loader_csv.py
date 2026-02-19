# data_loader_csv.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import torch
from model_core.config import ModelConfig
from model_core.factors import FeatureEngineer
from utils import cuda_snapshot, dict_tensors_snapshot, tensor_nbytes, _fmt_bytes


@dataclass
class CsvLoaderConfig:
    csv_paths: list[str]              # 可以一个或多个 csv（每个文件可包含多个symbol，也可单symbol）
    device: str = "cpu"               # "cpu" or "cuda"
    max_symbols: int | None = None
    tz_utc: bool = True               # 你的 open_time 看起来是带 +00:00 的
    liquidity_mode: str = "quote_volume"  # "quote_volume" or "constant"
    liquidity_constant: float = 1e9
    fdv_constant: float = 1e12        # 没有fdv就给常数，避免除0/NaN


class CsvCryptoDataLoader:
    """
    目标：输出与原 CryptoDataLoader 对齐的三个属性：
      - raw_data_cache: dict[str, torch.Tensor]  每个字段 [N, T]
      - feat_tensor: torch.Tensor                [N, F=6, T]
      - target_ret: torch.Tensor                 [N, T]
    """
    def __init__(self, cfg: CsvLoaderConfig):
        self.cfg = cfg
        self.raw_data_cache: dict[str, torch.Tensor] = {}
        self.feat_tensor: torch.Tensor | None = None
        self.target_ret: torch.Tensor | None = None
        self.symbols: list[str] = []

    def load_data(self):
        df = self._read_and_concat(self.cfg.csv_paths)
        if "symbol" not in df.columns:
            df["symbol"] = "ETHUSDT"

        '''torch.cuda.reset_peak_memory_stats(ModelConfig.DEVICE.index or 0)
        cuda_snapshot("load_data::start", ModelConfig.DEVICE)'''

        # ---- 标准化列名/生成 time/address ----
        # 用 symbol 代替 address（对中心化交易所足够唯一）
        df = df.rename(columns={"symbol": "address", "open_time": "time"})
        need_cols = ["time", "address", "open", "high", "low", "close", "volume"]
        missing = [c for c in need_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # time 解析：你的 open_time 多半是 '2022-12-31 17:48:00+00:00'
        df["time"] = pd.to_datetime(df["time"], utc=self.cfg.tz_utc, errors="coerce")
        if df["time"].isna().any():
            bad = df[df["time"].isna()].head(5)
            raise ValueError(f"Bad time parse rows (showing 5):\n{bad}")

        # 为了 pivot 稳定，把 tz 去掉（与原项目一致：naive time）
        df["time"] = df["time"].dt.tz_localize(None)

        # 转数值
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # ---- 补 liquidity/fdv ----
        if self.cfg.liquidity_mode == "quote_volume" and "quote_volume" in df.columns:
            # quote_volume 大多是以计价币计的成交额，拿来当 liquidity proxy（粗糙但可用）
            df["liquidity"] = pd.to_numeric(df["quote_volume"], errors="coerce")
            df["liquidity"] = df["liquidity"].fillna(0.0)
            # 量纲可能很大/很小，做个下限避免除0
            df["liquidity"] = df["liquidity"].clip(lower=1.0)
        else:
            df["liquidity"] = float(self.cfg.liquidity_constant)

        df["fdv"] = float(self.cfg.fdv_constant)

        # ---- 选symbol数量（可选）----
        sym_list = sorted(df["address"].unique().tolist())
        if self.cfg.max_symbols is not None:
            sym_list = sym_list[: self.cfg.max_symbols]
        df = df[df["address"].isin(sym_list)].copy()
        self.symbols = sym_list

        # ---- 去重排序（非常重要）----
        df = df.sort_values(["time", "address"])
        df = df.drop_duplicates(["time", "address"], keep="last")

        # ---- pivot → tensor [N, T] ----
        raw = {}
        for col in ["open", "high", "low", "close", "volume", "liquidity", "fdv"]:
            pv = df.pivot(index="time", columns="address", values=col)

            # 对齐缺失：与原代码一致（ffill + fillna(0)）
            pv = pv.ffill().fillna(0.0)

            # shape: [T, N] -> [N, T]
            tens = torch.tensor(pv.values.T, dtype=torch.float32, device=self.cfg.device)
            raw[col] = tens

        self.raw_data_cache = raw

        '''df_mem = int(df.memory_usage(deep=True).sum())
        print(f"[DF] rows={len(df):,} cols={len(df.columns)} df_mem={_fmt_bytes(df_mem)}")
        print(f"[DF] columns={list(df.columns)}")
        cuda_snapshot("load_data::after_df", ModelConfig.DEVICE)
        dict_tensors_snapshot("raw_data_cache", self.raw_data_cache, device=ModelConfig.DEVICE)'''

        # ---- 生成 6维特征 ----
        self.feat_tensor = FeatureEngineer.compute_features(self.raw_data_cache)
        '''cuda_snapshot("after_feat_tensor", ModelConfig.DEVICE,
                      extra=f"feat_shape={tuple(self.feat_tensor.shape)} size={_fmt_bytes(tensor_nbytes(self.feat_tensor))}")'''
        # ---- 生成 target_ret（与原 loader 类似：open-to-open）----
        open_ = self.raw_data_cache["open"]
        # 用 shift，不用 roll（避免 wrap-around）
        open_t1 = torch.cat([open_[:, 1:], open_[:, -1:]], dim=1)   # 近似：最后一列重复
        open_t2 = torch.cat([open_[:, 2:], open_[:, -1:], open_[:, -1:]], dim=1)
        target = torch.log((open_t2 + 1e-9) / (open_t1 + 1e-9))
        # 最后两列无意义，置0
        if target.shape[1] >= 2:
            target[:, -2:] = 0.0
        self.target_ret = target

        # cuda_snapshot("after_target_ret", ModelConfig.DEVICE, extra=f"target_shape={tuple(self.target_ret.shape)} size={_fmt_bytes(tensor_nbytes(self.target_ret))}")

        # print(f"Data Ready. Shape: {self.feat_tensor.shape}")
        return self
        # end load_data

    def _read_and_concat(self, paths: list[str]) -> pd.DataFrame:
        frames = []
        for p in paths:
            pth = Path(p)
            if not pth.exists():
                raise FileNotFoundError(str(pth))
            df = pd.read_csv(pth)
            frames.append(df)
        return pd.concat(frames, ignore_index=True)