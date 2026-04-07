# data_loader_csv.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from model_core.factors import FeatureEngineer


@dataclass
class CsvLoaderConfig:
    csv_paths: list[str]              # 可以一个或多个 csv
    device: str = "cpu"               # "cpu" or "cuda"
    max_symbols: int | None = None    # 旧模式用：按符号 pivot（向后兼容）
    lookback_bars: int = 0            # 每个 segment 前 N 根 K 线仅用于指标预热，不参与交易
    tz_utc: bool = True
    segment_mode: bool = False        # True = 按 segment_id 串联（新模式），False = 按 symbol pivot（旧模式）


class CsvCryptoDataLoader:
    """
    两种模式:
      旧模式 (segment_mode=False):
        按 symbol pivot → [N, T] 张量。与原 CryptoDataLoader 对齐。
      新模式 (segment_mode=True):
        按 segment_id 逐段计算特征后串联 → [1, F, T_total]。
        内存占用可降低数十倍到上百倍（消除稀疏 pivot 膨胀）。
    """
    def __init__(self, cfg: CsvLoaderConfig):
        self.cfg = cfg
        self.raw_data_cache: dict[str, torch.Tensor] = {}
        self.feat_tensor: torch.Tensor | None = None
        self.target_ret: torch.Tensor | None = None
        self.symbols: list[str] = []

        # ---- 新模式专用属性 ----
        self.n_segments: int = 0
        self.segment_ids: torch.Tensor | None = None      # [T_total] 每个时间步所属 segment index
        self.tradeable_mask: torch.Tensor | None = None    # [1, T_total] bool
        self.segment_starts: list[int] = []
        self.segment_trade_starts: list[int] = []
        self.segment_ends: list[int] = []

    # ------------------------------------------------------------------ #
    #                           公共入口                                   #
    # ------------------------------------------------------------------ #
    def load_data(self):
        if self.cfg.segment_mode:
            return self._load_segment_mode()
        return self._load_pivot_mode()

    # ------------------------------------------------------------------ #
    #        新模式：按 segment 串联，[1, F, T_total]                       #
    # ------------------------------------------------------------------ #
    def _load_segment_mode(self):
        df = self._read_and_concat(self.cfg.csv_paths)

        if "segment_id" not in df.columns:
            df["segment_id"] = 0

        # 时间列解析
        if "open_time" in df.columns:
            time_col = "open_time"
        elif "time" in df.columns:
            time_col = "time"
        else:
            raise ValueError("CSV 中缺少 open_time 或 time 列")

        df[time_col] = pd.to_datetime(df[time_col], utc=self.cfg.tz_utc, errors="coerce")

        # 数值列
        for c in ["open", "high", "low", "close", "volume"]:
            if c not in df.columns:
                raise ValueError(f"Missing required column: {c}")
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # 逐 segment 处理
        segments = sorted(df["segment_id"].unique())
        self.n_segments = len(segments)

        all_raw = {col: [] for col in ["open", "high", "low", "close", "volume"]}
        all_feat: list[torch.Tensor] = []
        all_target: list[torch.Tensor] = []
        seg_id_parts: list[torch.Tensor] = []

        offset = 0
        lookback = self.cfg.lookback_bars

        for seg_idx, seg_id in enumerate(segments):
            seg_df = df[df["segment_id"] == seg_id].sort_values(time_col).reset_index(drop=True)
            T_seg = len(seg_df)
            if T_seg == 0:
                continue

            self.segment_starts.append(offset)
            self.segment_trade_starts.append(offset + min(lookback, T_seg))
            self.segment_ends.append(offset + T_seg)

            # 构建 per-segment raw tensors [1, T_seg]
            seg_raw: dict[str, torch.Tensor] = {}
            for col in ["open", "high", "low", "close", "volume"]:
                vals = seg_df[col].values.astype(np.float32)
                seg_raw[col] = torch.tensor(vals, dtype=torch.float32).unsqueeze(0)  # [1, T_seg]

            # 逐段计算特征 → [1, F, T_seg]（滚动窗口/shift 自然隔离在段内）
            seg_feat = FeatureEngineer.compute_features(seg_raw)

            # 逐段计算 target_ret（open-to-open）→ [1, T_seg]
            open_ = seg_raw["open"]
            open_t1 = torch.cat([open_[:, 1:], open_[:, -1:]], dim=1)
            open_t2 = torch.cat([open_[:, 2:], open_[:, -1:], open_[:, -1:]], dim=1)
            target = torch.log((open_t2 + 1e-9) / (open_t1 + 1e-9))
            if target.shape[1] >= 2:
                target[:, -2:] = 0.0

            for col in ["open", "high", "low", "close", "volume"]:
                all_raw[col].append(seg_raw[col])
            all_feat.append(seg_feat)
            all_target.append(target)
            seg_id_parts.append(torch.full((T_seg,), seg_idx, dtype=torch.long))

            offset += T_seg

        if not all_feat:
            raise ValueError("没有任何有效的 segment 数据")

        T_total = offset
        dev = self.cfg.device

        # 拼接并移至目标设备
        for col in ["open", "high", "low", "close", "volume"]:
            self.raw_data_cache[col] = torch.cat(all_raw[col], dim=1).to(dev)   # [1, T_total]
        self.feat_tensor = torch.cat(all_feat, dim=2).to(dev)                   # [1, F, T_total]
        self.target_ret = torch.cat(all_target, dim=1).to(dev)                  # [1, T_total]

        # segment_ids: [T_total] 映射每个时间步 → segment index（用于 scatter_add 聚合）
        self.segment_ids = torch.cat(seg_id_parts).to(dev)

        # tradeable_mask: [1, T_total]
        mask = torch.zeros(1, T_total, dtype=torch.bool)
        for i in range(len(self.segment_starts)):
            ts = self.segment_trade_starts[i]
            te = self.segment_ends[i]
            # 最后 2 根 K 线的 target_ret 无效，不交易
            mask[0, ts:max(ts, te - 2)] = True
        self.tradeable_mask = mask.to(dev)

        print(f"[SegmentLoader] {self.n_segments} segments, T_total={T_total}, "
              f"tradeable={int(self.tradeable_mask.sum())}, "
              f"feat_shape={tuple(self.feat_tensor.shape)}")
        return self

    # ------------------------------------------------------------------ #
    #        旧模式：按 symbol pivot → [N, T]（向后兼容）                    #
    # ------------------------------------------------------------------ #
    def _load_pivot_mode(self):
        df = self._read_and_concat(self.cfg.csv_paths)
        if "symbol" not in df.columns:
            df["symbol"] = "ETHUSDT"

        df = df.rename(columns={"symbol": "address", "open_time": "time"})
        need_cols = ["time", "address", "open", "high", "low", "close", "volume"]
        missing = [c for c in need_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df["time"] = pd.to_datetime(df["time"], utc=self.cfg.tz_utc, errors="coerce")
        if df["time"].isna().any():
            bad = df[df["time"].isna()].head(5)
            raise ValueError(f"Bad time parse rows (showing 5):\n{bad}")
        df["time"] = df["time"].dt.tz_localize(None)

        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        sym_list = sorted(df["address"].unique().tolist())
        if self.cfg.max_symbols is not None:
            sym_list = sym_list[: self.cfg.max_symbols]
        df = df[df["address"].isin(sym_list)].copy()
        self.symbols = sym_list

        df = df.sort_values(["time", "address"])
        df = df.drop_duplicates(["time", "address"], keep="last")

        raw = {}
        for col in ["open", "high", "low", "close", "volume"]:
            pv = df.pivot(index="time", columns="address", values=col)
            pv = pv.ffill().fillna(0.0)
            tens = torch.tensor(pv.values.T, dtype=torch.float32, device=self.cfg.device)
            raw[col] = tens

        self.raw_data_cache = raw
        self.feat_tensor = FeatureEngineer.compute_features(self.raw_data_cache)

        open_ = self.raw_data_cache["open"]
        open_t1 = torch.cat([open_[:, 1:], open_[:, -1:]], dim=1)
        open_t2 = torch.cat([open_[:, 2:], open_[:, -1:], open_[:, -1:]], dim=1)
        target = torch.log((open_t2 + 1e-9) / (open_t1 + 1e-9))
        if target.shape[1] >= 2:
            target[:, -2:] = 0.0
        self.target_ret = target

        return self

    # ------------------------------------------------------------------ #
    def _read_and_concat(self, paths: list[str]) -> pd.DataFrame:
        frames = []
        for p in paths:
            pth = Path(p)
            if not pth.exists():
                raise FileNotFoundError(str(pth))
            df = pd.read_csv(pth)
            frames.append(df)
        return pd.concat(frames, ignore_index=True)
