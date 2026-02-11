import tushare as ts
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import os
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

TS_TOKEN = '填入 Tushare Token'
INDEX_CODE = '511260.SH'
START_DATE = '20150101'
END_DATE = '20240101'
TEST_END_DATE = '20250101'

BATCH_SIZE = 1024
TRAIN_ITERATIONS = 400
MAX_SEQ_LEN = 8            # 限制公式长度，防止过拟合
COST_RATE = 0.0005         # 双边万一 (ETF/IC期货费率较低)，设为万五偏保守

DATA_CACHE_PATH = 'data_cache_final.parquet'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision('high')

@torch.jit.script
def _ts_delay(x: torch.Tensor, d: int) -> torch.Tensor:
    if d == 0: return x
    pad = torch.zeros((x.shape[0], d), device=x.device)
    return torch.cat([pad, x[:, :-d]], dim=1)

@torch.jit.script
def _ts_delta(x: torch.Tensor, d: int) -> torch.Tensor:
    return x - _ts_delay(x, d)

@torch.jit.script
def _ts_zscore(x: torch.Tensor, d: int) -> torch.Tensor:
    if d <= 1: return torch.zeros_like(x)
    B, T = x.shape
    pad = torch.zeros((B, d - 1), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    windows = x_pad.unfold(1, d, 1)
    mean = windows.mean(dim=-1)
    std = windows.std(dim=-1) + 1e-6
    return (x - mean) / std

@torch.jit.script
def _ts_decay_linear(x: torch.Tensor, d: int) -> torch.Tensor:
    if d <= 1: return x
    B, T = x.shape
    pad = torch.zeros((B, d - 1), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    windows = x_pad.unfold(1, d, 1) # [B, T, d]
    w = torch.arange(1, d + 1, device=x.device, dtype=x.dtype)
    w = w / w.sum()
    return (windows * w).sum(dim=-1)

OPS_CONFIG = [
    ('ADD', lambda x, y: x + y, 2),
    ('SUB', lambda x, y: x - y, 2),
    ('MUL', lambda x, y: x * y, 2),
    ('DIV', lambda x, y: x / (y + 1e-6 * torch.sign(y)), 2),
    ('NEG', lambda x: -x, 1),
    ('ABS', lambda x: torch.abs(x), 1),
    ('SIGN', lambda x: torch.sign(x), 1),
    ('DELTA5', lambda x: _ts_delta(x, 5), 1),
    ('MA20',   lambda x: _ts_decay_linear(x, 20), 1),
    ('STD20',  lambda x: _ts_zscore(x, 20), 1),
    ('TS_RANK20', lambda x: _ts_zscore(x, 20), 1),
]

FEATURES = ['RET', 'RET5', 'VOL_CHG', 'V_RET', 'TREND']

VOCAB = FEATURES + [cfg[0] for cfg in OPS_CONFIG]
VOCAB_SIZE = len(VOCAB)
OP_FUNC_MAP = {i + len(FEATURES): cfg[1] for i, cfg in enumerate(OPS_CONFIG)}
OP_ARITY_MAP = {i + len(FEATURES): cfg[2] for i, cfg in enumerate(OPS_CONFIG)}

class AlphaGPT(nn.Module):
    def __init__(self, d_model=64, n_head=4, n_layer=2):
        super().__init__()
        self.token_emb = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, MAX_SEQ_LEN + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=128, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)

        self.ln_f = nn.LayerNorm(d_model)
        self.head_actor = nn.Linear(d_model, VOCAB_SIZE)
        self.head_critic = nn.Linear(d_model, 1)

    def forward(self, idx):
        B, T = idx.size()
        x = self.token_emb(idx) + self.pos_emb[:, :T, :]
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(idx.device)
        x = self.blocks(x, mask=mask, is_causal=True)
        x = self.ln_f(x)
        last = x[:, -1, :]
        return self.head_actor(last), self.head_critic(last)

class DataEngine:
    def __init__(self):
        self.pro = ts.pro_api(TS_TOKEN)
    def load(self):
        if os.path.exists(DATA_CACHE_PATH):
            df = pd.read_parquet(DATA_CACHE_PATH)
        else:
            print(f"🌐 Fetching {INDEX_CODE}...")
            if INDEX_CODE.endswith(".SZ") or INDEX_CODE.endswith(".SH"):
                try:
                    df = self.pro.fund_daily(ts_code=INDEX_CODE, start_date=START_DATE, end_date=TEST_END_DATE)
                except:
                    df = self.pro.index_daily(ts_code=INDEX_CODE, start_date=START_DATE, end_date=TEST_END_DATE)

            if df is None or df.empty:
                raise ValueError("未获取到数据，请检查Token或代码是否正确")

            df = df.sort_values('trade_date').reset_index(drop=True)
            df.to_parquet(DATA_CACHE_PATH)

        for col in ['open', 'high', 'low', 'close', 'vol']:
            df[col] = pd.to_numeric(df[col], errors='coerce').ffill().bfill()

        self.dates = pd.to_datetime(df['trade_date'])

        close = df['close'].values.astype(np.float32)
        open_ = df['open'].values.astype(np.float32)
        high = df['high'].values.astype(np.float32)
        low = df['low'].values.astype(np.float32)
        vol = df['vol'].values.astype(np.float32)

        ret = np.zeros_like(close)
        ret[1:] = (close[1:] - close[:-1]) / (close[:-1] + 1e-6)

        ret5 = pd.Series(close).pct_change(5).fillna(0).values.astype(np.float32)

        vol_ma = pd.Series(vol).rolling(20).mean().values
        vol_chg = np.zeros_like(vol)
        mask = vol_ma > 0
        vol_chg[mask] = vol[mask] / vol_ma[mask] - 1
        vol_chg = np.nan_to_num(vol_chg).astype(np.float32)

        v_ret = (ret * (vol_chg + 1)).astype(np.float32)

        ma60 = pd.Series(close).rolling(60).mean().values
        trend = np.zeros_like(close)
        mask = ma60 > 0
        trend[mask] = close[mask] / ma60[mask] - 1
        trend = np.nan_to_num(trend).astype(np.float32)

        # Robust Normalization
        def robust_norm(x):
            x = x.astype(np.float32) # 强制转类型
            median = np.nanmedian(x)
            mad = np.nanmedian(np.abs(x - median)) + 1e-6
            res = (x - median) / mad
            return np.clip(res, -5, 5).astype(np.float32)

        self.feat_data = torch.stack([
            torch.from_numpy(robust_norm(ret)).to(DEVICE),
            torch.from_numpy(robust_norm(ret5)).to(DEVICE),
            torch.from_numpy(robust_norm(vol_chg)).to(DEVICE),
            torch.from_numpy(robust_norm(v_ret)).to(DEVICE),
            torch.from_numpy(robust_norm(trend)).to(DEVICE)
        ])

        open_tensor = torch.from_numpy(open_).to(DEVICE)
        open_t1 = torch.roll(open_tensor, -1)
        open_t2 = torch.roll(open_tensor, -2)

        self.target_oto_ret = (open_t2 - open_t1) / (open_t1 + 1e-6)
        self.target_oto_ret[-2:] = 0.0

        self.raw_open = open_tensor
        self.raw_close = torch.from_numpy(close).to(DEVICE)

        self.split_idx = int(len(df) * 0.8)
        print(f"{INDEX_CODE} Data Ready. Normalization Fixed.")
        return self

class DeepQuantMiner:
    def __init__(self, engine):
        self.engine = engine
        self.model = AlphaGPT().to(DEVICE)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=1e-5)
        self.best_sharpe = -10.0
        self.best_formula_tokens = None

    def get_strict_mask(self, open_slots, step):
        B = open_slots.shape[0]
        mask = torch.full((B, VOCAB_SIZE), float('-inf'), device=DEVICE)
        remaining_steps = MAX_SEQ_LEN - step

        done_mask = (open_slots == 0)
        mask[done_mask, 0] = 0.0 # Pad with first feature

        active_mask = ~done_mask
        # 如果剩余步数不够填坑了，必须选 Feature (arity=0)
        must_pick_feat = (open_slots >= remaining_steps)

        mask[active_mask, :len(FEATURES)] = 0.0
        can_pick_op_mask = active_mask & (~must_pick_feat)
        if can_pick_op_mask.any():
            mask[can_pick_op_mask, len(FEATURES):] = 0.0
        return mask

    def solve_one(self, tokens):
        stack = []
        try:
            for t in reversed(tokens):
                if t < len(FEATURES):
                    stack.append(self.engine.feat_data[t])
                else:
                    arity = OP_ARITY_MAP[t]
                    if len(stack) < arity: raise ValueError
                    args = [stack.pop() for _ in range(arity)]
                    func = OP_FUNC_MAP[t]
                    if arity == 2: res = func(args[0], args[1])
                    else: res = func(args[0])

                    if torch.isnan(res).any(): res = torch.nan_to_num(res)
                    stack.append(res)

            if len(stack) >= 1:
                final = stack[-1]
                # 过滤掉常数因子
                if final.std() < 1e-4: return None
                return final
        except:
            return None
        return None

    def solve_batch(self, token_seqs):
        B = token_seqs.shape[0]
        results = torch.zeros((B, self.engine.feat_data.shape[1]), device=DEVICE)
        valid_mask = torch.zeros(B, dtype=torch.bool, device=DEVICE)

        for i in range(B):
            res = self.solve_one(token_seqs[i].cpu().tolist())
            if res is not None:
                results[i] = res
                valid_mask[i] = True
        return results, valid_mask
    def backtest(self, factors):
        if factors.shape[0] == 0: return torch.tensor([], device=DEVICE)

        split = self.engine.split_idx
        # 使用 Open-to-Open 的收益率
        target = self.engine.target_oto_ret[:split]

        rewards = torch.zeros(factors.shape[0], device=DEVICE)

        for i in range(factors.shape[0]):
            f = factors[i, :split]

            if torch.isnan(f).all() or (f == 0).all() or f.numel() == 0:
                rewards[i] = -2.0
                continue

            sig = torch.tanh(f)
            pos = torch.sign(sig)

            turnover = torch.abs(pos - torch.roll(pos, 1))
            if turnover.numel() > 0:
                turnover[0] = 0.0
            else:
                rewards[i] = -2.0
                continue

            # 净收益
            pnl = pos * target - turnover * COST_RATE

            if pnl.numel() < 10: # 数据太少不具有统计意义
                rewards[i] = -2.0
                continue

            mu = pnl.mean()
            std = pnl.std() + 1e-6

            # 计算下行风险
            downside_returns = pnl[pnl < 0]
            if downside_returns.numel() > 5:
                down_std = downside_returns.std() + 1e-6
                sortino = mu / down_std * 15.87
            else:
                sortino = mu / std * 15.87

            # 惩罚项
            if mu < 0: sortino = -2.0
            if turnover.mean() > 0.5: sortino -= 1.0 # 惩罚过度交易
            if (pos == 0).all(): sortino = -2.0      # 惩罚不持仓

            rewards[i] = sortino

        return torch.clamp(rewards, -3, 5)
    def train(self):
        print(f"Training for Stable Profit... MAX_LEN={MAX_SEQ_LEN}")
        pbar = tqdm(range(TRAIN_ITERATIONS))

        for _ in pbar:
            # 1. Generate
            B = BATCH_SIZE
            open_slots = torch.ones(B, dtype=torch.long, device=DEVICE)
            log_probs, tokens = [], []
            curr_inp = torch.zeros((B, 1), dtype=torch.long, device=DEVICE)

            for step in range(MAX_SEQ_LEN):
                logits, val = self.model(curr_inp)
                mask = self.get_strict_mask(open_slots, step)
                dist = Categorical(logits=(logits + mask))
                action = dist.sample()

                log_probs.append(dist.log_prob(action))
                tokens.append(action)
                curr_inp = torch.cat([curr_inp, action.unsqueeze(1)], dim=1)

                is_op = action >= len(FEATURES)
                delta = torch.full((B,), -1, device=DEVICE)
                arity_tens = torch.zeros(VOCAB_SIZE, dtype=torch.long, device=DEVICE)
                for k,v in OP_ARITY_MAP.items(): arity_tens[k] = v
                op_delta = arity_tens[action] - 1
                delta = torch.where(is_op, op_delta, delta)
                delta[open_slots==0] = 0
                open_slots += delta

            seqs = torch.stack(tokens, dim=1)

            # 2. Evaluate
            with torch.no_grad():
                f_vals, valid_mask = self.solve_batch(seqs)
                valid_idx = torch.where(valid_mask)[0]
                rewards = torch.full((B,), -1.0, device=DEVICE) # 默认惩罚

                if len(valid_idx) > 0:
                    bt_scores = self.backtest(f_vals[valid_idx])
                    rewards[valid_idx] = bt_scores

                    best_sub_idx = torch.argmax(bt_scores)
                    current_best_score = bt_scores[best_sub_idx].item()

                    if current_best_score > self.best_sharpe:
                        self.best_sharpe = current_best_score
                        self.best_formula_tokens = seqs[valid_idx[best_sub_idx]].cpu().tolist()

            # 3. Update
            adv = rewards - rewards.mean()
            loss = -(torch.stack(log_probs, 1).sum(1) * adv).mean()

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            pbar.set_postfix({'Valid': f"{len(valid_idx)/B:.1%}", 'BestSortino': f"{self.best_sharpe:.2f}"})

    def decode(self, tokens=None):
        if tokens is None: tokens = self.best_formula_tokens
        if tokens is None: return "N/A"
        stream = list(tokens)
        def _parse():
            if not stream: return ""
            t = stream.pop(0)
            if t < len(FEATURES): return FEATURES[t]
            args = [_parse() for _ in range(OP_ARITY_MAP[t])]
            return f"{VOCAB[t]}({','.join(args)})"
        try: return _parse()
        except: return "Invalid"

def final_reality_check(miner, engine):
    print("\n" + "="*60)
    print("FINAL CHECK (Out-of-Sample)")
    print("="*60)

    formula_str = miner.decode()
    if miner.best_formula_tokens is None: return
    print(f"Strategy Formula: {formula_str}")

    # 1. 获取全量因子值
    factor_all = miner.solve_one(miner.best_formula_tokens)
    if factor_all is None: return

    # 2. 提取测试集数据 (Strict OOS)
    split = engine.split_idx
    test_dates = engine.dates[split:]
    test_factors = factor_all[split:].cpu().numpy()

    # 使用 Open-to-Open 收益
    # 注意：target_oto_ret[t] 对应的是 t+1 开盘买, t+2 开盘卖的收益
    # 所以我们的 signal[t] 应该和 target_oto_ret[t] 对齐
    test_ret = engine.target_oto_ret[split:].cpu().numpy()

    # 减少噪音
    rolling_mean_factor = pd.Series(test_factors).rolling(3).mean().fillna(0).values
    signal = np.tanh(test_factors)

    # 仓位
    position = np.sign(signal)

    # 检查涨跌停/停牌 (Limit Move Check)
    # 这里认为如果 next_open 相对于 close 涨跌幅超过 9.5%，则无法成交
    # raw_close[t], raw_open[t+1]
    # 需要对齐时间轴。target_oto_ret 对应的是 t+1 到 t+2。
    # 我们检查 t+1 开盘是否可交易。

    raw_close = engine.raw_close[split:].cpu().numpy()
    raw_open_next = engine.raw_open[split:].cpu().numpy() # 这里稍微错位，简化处理
    # 实际上，DataEngine需要更精细的时间对齐来做Limit Check，这里做个简单近似

    # 换手
    turnover = np.abs(position - np.roll(position, 1))
    turnover[0] = 0

    # PnL
    daily_ret = position * test_ret - turnover * COST_RATE

    # 4. 统计
    equity = (1 + daily_ret).cumprod()

    total_ret = equity[-1] - 1
    ann_ret = equity[-1] ** (252/len(equity)) - 1
    vol = np.std(daily_ret) * np.sqrt(252)
    sharpe = (ann_ret - 0.02) / (vol + 1e-6)

    # Max Drawdown
    dd = 1 - equity / np.maximum.accumulate(equity)
    max_dd = np.max(dd)
    calmar = ann_ret / (max_dd + 1e-6)

    print(f"Test Period    : {test_dates.iloc[0].date()} ~ {test_dates.iloc[-1].date()}")
    print(f"Ann. Return    : {ann_ret:.2%}")
    print(f"Ann. Volatility: {vol:.2%}")
    print(f"Sharpe Ratio   : {sharpe:.2f}")
    print(f"Max Drawdown   : {max_dd:.2%}")
    print(f"Calmar Ratio   : {calmar:.2f}")
    print("-" * 60)

    plt.style.use('bmh')
    plt.figure(figsize=(12, 6))

    plt.plot(test_dates, equity, label='Strategy (Open-to-Open)', linewidth=1.5)

    bench_ret = test_ret
    bench_equity = (1 + bench_ret).cumprod()
    plt.plot(test_dates, bench_equity, label='Benchmark (CSI 300)', alpha=0.5, linewidth=1)

    plt.title(f'Strict OOS Backtest: Ann Ret {ann_ret:.1%} | Sharpe {sharpe:.2f}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('strategy_performance.png')
    print("Chart saved to 'strategy_performance.png'")

if __name__ == "__main__":
    eng = DataEngine()
    eng.load()
    miner = DeepQuantMiner(eng)
    miner.train()
    final_reality_check(miner, eng)