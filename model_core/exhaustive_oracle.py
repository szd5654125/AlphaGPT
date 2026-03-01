import argparse
import heapq
import multiprocessing as mp
from typing import Iterable, List, Optional, Set, Tuple
import time
import torch
from config.general_config import ModelConfig
from model_core.ops import OPS_CONFIG
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest
from model_core.data_loader_csv import CsvCryptoDataLoader, CsvLoaderConfig


def build_arity_vec(device: torch.device) -> torch.Tensor:
    feat_offset = FeatureEngineer.INPUT_DIM
    vocab_size = feat_offset + len(OPS_CONFIG)
    arity_vec = torch.zeros(vocab_size, dtype=torch.long, device=device)
    for j, (_, _, arity) in enumerate(OPS_CONFIG):
        arity_vec[feat_offset + j] = arity
    return arity_vec


def enumerate_rpn(
    max_len: int,
    min_len: int,
    feat_ids: List[int],
    op_ids: List[int],
    arity_vec: torch.Tensor,
) -> Iterable[Tuple[List[int], int]]:
    """
    Yield (formula_tokens, ops_count) for any prefix that can legally terminate (depth==1 and len>=min_len),
    while exploring all sequences up to max_len under stack constraints.
    """
    # max_reduce = max(arity-1) for ops
    if op_ids:
        max_reduce = max((int(arity_vec[t].item()) - 1) for t in op_ids)
    else:
        max_reduce = 0

    tokens = feat_ids + op_ids

    seq: List[int] = []

    def rec(step: int, depth: int, ops_cnt: int):
        # step: current length
        if step >= min_len and depth == 1:
            yield (seq.copy(), ops_cnt)

        if step == max_len:
            return

        r_after = max_len - (step + 1)

        for tok in tokens:
            arity = int(arity_vec[tok].item())
            if arity > 0 and depth < arity:
                continue  # underflow

            depth_after = depth + 1 if arity == 0 else depth - arity + 1

            # finishable pruning (same spirit as your strict mask) :contentReference[oaicite:8]{index=8}
            if r_after == 0:
                if depth_after != 1:
                    continue
            elif max_reduce <= 0:
                if depth_after != 1:
                    continue
            else:
                if depth_after > max_reduce * r_after + 1:
                    continue

            seq.append(tok)
            yield from rec(step + 1, depth_after, ops_cnt + (1 if arity > 0 else 0))
            seq.pop()

    yield from rec(step=0, depth=0, ops_cnt=0)


def evaluate_formula_subset(work_item):
    (
        csv_path,
        device_name,
        max_len,
        min_len,
        feat_ids,
        op_ids,
        arity_values,
        topk,
        first_tokens,
    ) = work_item

    device = torch.device(device_name)
    cfg = CsvLoaderConfig(
        csv_paths=[csv_path],
        device=str(device),
        max_symbols=50,
    )
    loader = CsvCryptoDataLoader(cfg).load_data()
    vm = StackVM()
    bt = MemeBacktest()

    arity_vec = torch.tensor(arity_values, dtype=torch.long, device=device)
    lam_ops = float(ModelConfig.OPS_PENALTY_LAMBDA)

    heap: List[Tuple[float, float, float, int, List[int]]] = []
    n_eval = 0
    n_generated = 0

    for formula, ops_cnt in enumerate_rpn(
        max_len=max_len,
        min_len=min_len,
        feat_ids=feat_ids,
        op_ids=op_ids,
        arity_vec=arity_vec,
        first_token_filter=set(first_tokens),
    ):
        n_generated += 1
        res, info = vm.execute(formula, loader.feat_tensor)
        if res is None:
            continue
        if res.std() < 1e-4:
            continue

        best_local = None
        for thr in ModelConfig.THRESH_BINS:
            score, ret_val, details = bt.evaluate(res, loader.raw_data_cache, loader.target_ret, float(thr))
            score_f = float(score.item())
            reward_f = score_f - lam_ops * float(ops_cnt)
            cand = (reward_f, score_f, float(thr), ops_cnt, formula)
            if best_local is None or cand[0] > best_local[0]:
                best_local = cand

        if best_local is None:
            continue

        n_eval += 1
        if len(heap) < topk:
            heapq.heappush(heap, best_local)
        else:
            if best_local[0] > heap[0][0]:
                heapq.heapreplace(heap, best_local)

    return heap, n_generated, n_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="data/futures_um_monthly_klines_ETHUSDT_5m_0_53.csv")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--max-len", type=int, default=6)
    ap.add_argument("--min-len", type=int, default=4)
    ap.add_argument("--k-feats", type=int, default=8, help="use first k features (0..k-1)")
    ap.add_argument("--ops", type=str, default="ADD,SUB,MUL,DIV, NEG, ABS,SIGN,GATE", help="comma-separated op names")
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()
    t0 = time.time()
    print("[stage] init device/config", flush=True)
    device = torch.device(args.device)
    cfg = CsvLoaderConfig(
        csv_paths=[args.csv],
        device=str(device),
        max_symbols=50,
    )
    print("[stage] loading csv data", flush=True)
    loader = CsvCryptoDataLoader(cfg).load_data()
    print("[stage] building vm/backtest context", flush=True)

    vm = StackVM()
    bt = MemeBacktest()

    arity_vec = build_arity_vec(device=device)
    feat_offset = FeatureEngineer.INPUT_DIM

    feat_ids = list(range(min(args.k_feats, feat_offset)))

    name_to_opid = {name: (feat_offset + i) for i, (name, _, _) in enumerate(OPS_CONFIG)}
    op_names = [s.strip() for s in args.ops.split(",") if s.strip()]
    op_ids = []
    for n in op_names:
        if n not in name_to_opid:
            raise ValueError(f"Unknown op name: {n}. Available: {list(name_to_opid.keys())}")
        op_ids.append(name_to_opid[n])

    # min-heap of (reward, score, thr, ops_cnt, formula)
    heap: List[Tuple[float, float, float, int, List[int]]] = []

    n_eval = 0
    n_generated = 0
    print("[stage] start exhaustive enumeration", flush=True)
    all_tokens = feat_ids + op_ids
    token_shards = [all_tokens[i::16] for i in range(16)]
    work_items = [
        (
            args.csv,
            args.device,
            args.max_len,
            args.min_len,
            feat_ids,
            op_ids,
            arity_vec.cpu().tolist(),
            args.topk,
            shard,
        )
        for shard in token_shards
        if shard
    ]
    print(f"[stage] multiprocessing with {len(work_items)} workers", flush=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(work_items)) as pool:
        partial_results = pool.map(evaluate_formula_subset, work_items)

    for partial_heap, local_generated, local_eval in partial_results:
        n_generated += local_generated
        n_eval += local_eval
        for cand in partial_heap:
            if len(heap) < args.topk:
                heapq.heappush(heap, cand)
            else:
                if cand[0] > heap[0][0]:
                    heapq.heapreplace(heap, cand)

    best = sorted(heap, key=lambda x: x[0], reverse=True)
    print(f"[stage] done in {time.time() - t0:.1f}s", flush=True)
    print(f"Generated candidates: {n_generated}")
    print(f"Evaluated candidates: {n_eval}")
    for i, (rew, score, thr, ops_cnt, formula) in enumerate(best, 1):
        print(f"[{i:02d}] reward={rew:.6f} score={score:.6f} thr={thr:.2f} ops={ops_cnt} formula={formula}")



if __name__ == "__main__":
    main()