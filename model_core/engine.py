import torch
from collections import deque
import hashlib
import random
from torch.distributions import Bernoulli
from torch.distributions import Categorical as CatDist
import torch.nn.functional as F
from tqdm import tqdm
import os
import json
import multiprocessing as mp
import queue as _q
import numpy as np
from model_core.factors import FeatureEngineer
from model_core.ops import OPS_CONFIG
from config.general_config import ModelConfig
from model_core.data_loader_csv import CsvCryptoDataLoader, CsvLoaderConfig
from model_core.alphagpt import AlphaGPT, NewtonSchulzLowRankDecay, StableRankMonitor
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest


class AlphaEngine:
    def __init__(self, use_lord_regularization=True, lord_decay_rate=1e-3, lord_num_iterations=5, seed=None,
                 device=None, output_dir=".",):
        """
        Initialize AlphaGPT training engine.
        
        Args:
            use_lord_regularization: Enable Low-Rank Decay (LoRD) regularization
            lord_decay_rate: Strength of LoRD regularization
            lord_num_iterations: Number of Newton-Schulz iterations per step
        """
        self.seed = ModelConfig.SEED if seed is None else int(seed)
        self.device = ModelConfig.DEVICE if device is None else torch.device(device)
        self.output_dir = output_dir
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        '''self.loader = CryptoDataLoader()
        self.loader.load_data()'''
        cfg = CsvLoaderConfig(
            csv_paths=["data/futures_um_monthly_klines_ETHUSDT_5m_0_53.csv"],  # 改成你的路径
            device=self.device,
            max_symbols=50,  # 你可以先小一点试跑
        )
        self.loader = CsvCryptoDataLoader(cfg).load_data()
        self.model = AlphaGPT().to(self.device)
        expected_vocab = FeatureEngineer.INPUT_DIM + len(OPS_CONFIG)
        if getattr(self.model, "vocab_size", None) != expected_vocab:
            raise ValueError(
                f"Vocab mismatch: model.vocab_size={self.model.vocab_size}, "
                f"but expected={expected_vocab} (FeatureEngineer.INPUT_DIM + len(OPS_CONFIG))"
            )
        
        # Standard optimizer
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        
        # Low-Rank Decay regularizer
        self.use_lord = use_lord_regularization
        if self.use_lord:
            self.lord_opt = NewtonSchulzLowRankDecay(
                self.model.named_parameters(),
                decay_rate=lord_decay_rate,
                num_iterations=lord_num_iterations,
                target_keywords=["q_proj", "k_proj", "attention", "qk_norm"]
            )
            self.rank_monitor = StableRankMonitor(
                self.model,
                target_keywords=["q_proj", "k_proj"]
            )
        else:
            self.lord_opt = None
            self.rank_monitor = None
        
        self.vm = StackVM()
        self.bt = MemeBacktest()
        
        self.best_score = -float('inf')
        self.best_raw_score = -float('inf')
        self.best_formula = None
        self.best_threshold = None
        self.training_history = {
            'step': [],
            'avg_reward': [],
            'best_score': [],
            'stable_rank': []
        }

        self.feat_offset = FeatureEngineer.INPUT_DIM
        self.vocab_size = self.feat_offset + len(OPS_CONFIG)
        # 每个 token 的 arity：feature=0，op=1/2/3...
        self.arity_vec = torch.zeros(self.vocab_size, dtype=torch.long, device=self.device)
        for j, (_, _, arity) in enumerate(OPS_CONFIG):
            self.arity_vec[self.feat_offset + j] = arity
        # 单步最大“降栈幅度”：max(arity-1)，用于判断“剩余步数是否还能收敛到 1”
        self.max_reduce = max((arity - 1 for (_, _, arity) in OPS_CONFIG), default=0)
        # ---- formula de-dup (exact token sequence) ----
        self._seen_global = set()
        self._seen_fifo = deque()
        self._seen_limit = int(getattr(ModelConfig, "DEDUP_GLOBAL_CACHE", 300000))
        # ---- canonical token de-dup (after simplify/canonicalize) ----
        self._seen_can_global = set()
        self._seen_can_fifo = deque()
        self._seen_can_limit = int(getattr(ModelConfig, "DEDUP_CANON_CACHE", self._seen_limit))
        self._dedup_max_tries = int(getattr(ModelConfig, "DEDUP_MAX_TRIES", 8))
        self._dedup_fail = 0
        # ---- semantic de-dup ----
        self._seen_sem_global = set()
        self._seen_sem_fifo = deque()
        self._seen_sem_limit = int(getattr(ModelConfig, "DEDUP_SEM_CACHE", 300000))
        self._use_sem_dedup = bool(getattr(ModelConfig, "DEDUP_USE_SEM", True))
        # op id -> (name, arity)
        self._op_meta = []
        for j, (name, _, arity) in enumerate(OPS_CONFIG):
            self._op_meta.append((name, arity))
        # canonicalization rules: conservative
        self._comm_ops = {"ADD", "MUL"}
        self._assoc_ops = {"ADD", "MUL"}  # 如果你不想 flatten，把这行设为空 set()
        self._idem_ops = {"ABS", "SIGN"}

    @staticmethod
    def _ast_equal(a, b):
        """Structural AST equality for small canonical rewrite checks."""
        return a == b

    def _key64(self, toks):
        # 8-byte digest -> python int
        h = hashlib.blake2b(digest_size=8)
        # vocab 很小，1 byte 足够；更通用点用 2 bytes
        for t in toks:
            h.update(int(t).to_bytes(2, "little", signed=False))
        return int.from_bytes(h.digest(), "little", signed=False)

    def _ast_hash64(self, node):
        """Hash canonical AST -> 64-bit int."""
        h = hashlib.blake2b(digest_size=8)
        def walk(n):
            tag = n[0]
            if tag == "F":
                h.update(b"F")
                h.update(int(n[1]).to_bytes(2, "little", signed=False))
                return
            # ("OP", name, children_tuple)
            h.update(b"O")
            name_b = n[1].encode("utf-8")
            h.update(len(name_b).to_bytes(1, "little"))
            h.update(name_b)
            kids = n[2]
            h.update(len(kids).to_bytes(1, "little"))
            h.update(b"[")
            for c in kids:
                walk(c)
            h.update(b"]")
        walk(node)
        return int.from_bytes(h.digest(), "little", signed=False)

    def _rpn_to_ast(self, toks):
        """RPN tokens -> AST node. Return None if malformed."""
        st = []
        for t in toks:
            if t < self.feat_offset:
                st.append(("F", int(t)))
                continue
            op_idx = int(t - self.feat_offset)
            if op_idx < 0 or op_idx >= len(self._op_meta):
                return None
            name, arity = self._op_meta[op_idx]
            if len(st) < arity:
                return None
            if arity == 1:
                x = st.pop()
                st.append(("OP", name, (x,)))
            elif arity == 2:
                b = st.pop()
                a = st.pop()
                st.append(("OP", name, (a, b)))
            elif arity == 3:
                c = st.pop()
                b = st.pop()
                a = st.pop()
                st.append(("OP", name, (a, b, c)))
            else:
                return None
        if len(st) != 1:
            return None
        return st[0]

    def _canon(self, node):
        """Canonicalize AST conservatively."""
        if node is None:
            return None
        if node[0] == "F":
            return node
        _, name, kids = node
        kids_c = tuple(self._canon(k) for k in kids)
        # NEG(NEG(x)) -> x
        if name == "NEG":
            x = kids_c[0]
            if x[0] == "OP" and x[1] == "NEG":
                return x[2][0]
            # NEG(SUB(a,b)) -> SUB(b,a)
            if x[0] == "OP" and x[1] == "SUB":
                a, b = x[2]
                return ("OP", "SUB", (b, a))
            return ("OP", "NEG", (x,))
        # idempotent: ABS(ABS(x)) -> ABS(x), SIGN(SIGN(x)) -> SIGN(x)
        if name in self._idem_ops:
            x = kids_c[0]
            if x[0] == "OP" and x[1] == name:
                return x
            # ABS(NEG(x)) == ABS(x)
            if name == "ABS" and x[0] == "OP" and x[1] == "NEG":
                return ("OP", "ABS", (x[2][0],))
            return ("OP", name, (x,))
        # SUB(x, NEG(y)) -> ADD(x, y)
        if name == "SUB" and len(kids_c) == 2:
            a, b = kids_c
            if b[0] == "OP" and b[1] == "NEG":
                return self._canon(("OP", "ADD", (a, b[2][0])))
            if a[0] == "OP" and a[1] == "NEG" and b[0] == "OP" and b[1] == "NEG":
                # SUB(NEG(a), NEG(b)) -> SUB(b, a)
                return ("OP", "SUB", (b[2][0], a[2][0]))
            return ("OP", "SUB", (a, b))
        # MUL sign-normalization: keep at most one top-level NEG.
        # Works for both binary and flattened n-ary MUL nodes.
        if name == "MUL":
            flat = []
            for k in kids_c:
                if (k[0] == "OP") and (k[1] == "MUL"):
                    flat.extend(list(k[2]))
                else:
                    flat.append(k)
            neg_cnt = 0
            stripped = []
            for k in flat:
                if (k[0] == "OP") and (k[1] == "NEG") and len(k[2]) == 1:
                    neg_cnt += 1
                    stripped.append(k[2][0])
                else:
                    stripped.append(k)
            base_mul = ("OP", "MUL", tuple(stripped))
            if neg_cnt % 2 == 1:
                return ("OP", "NEG", (self._canon(base_mul),))
            kids_c = tuple(stripped)
        # GATE(cond, x, x) -> x
        if name == "GATE" and len(kids_c) == 3:
            c, x, y = kids_c
            if self._ast_equal(x, y):
                return x
            return ("OP", "GATE", (c, x, y))
        # commutative + associative: flatten + sort
        if name in self._comm_ops:
            flat = []
            for k in kids_c:
                if (name in self._assoc_ops) and (k[0] == "OP") and (k[1] == name):
                    flat.extend(list(k[2]))
                else:
                    flat.append(k)
            # sort by structural hash for determinism
            flat_sorted = sorted(flat, key=lambda n: (self._ast_hash64(n), repr(n)))
            return ("OP", name, tuple(flat_sorted))
        # default: keep order
        return ("OP", name, kids_c)

    def _sem_key64(self, toks):
        ast = self._rpn_to_ast(toks)
        if ast is None:
            # fallback: treat as raw
            return self._key64(toks)
        can = self._canon(ast)
        return self._ast_hash64(can)

    def _ast_to_rpn(self, node):
        """AST -> canonical RPN tokens. Return None if node is malformed."""
        out = []

        def walk(n):
            if n is None:
                return False
            tag = n[0]
            if tag == "F":
                out.append(int(n[1]))
                return True
            if tag != "OP":
                return False
            _, name, kids = n
            for c in kids:
                if not walk(c):
                    return False
            for j, (op_name, op_arity) in enumerate(self._op_meta):
                if op_name != name:
                    continue
                tok = self.feat_offset + j
                # Canonical commutative flattening may produce n-ary ADD/MUL AST nodes.
                # Emit valid binary-RPN by applying the operator (len(kids)-1) times.
                if name in self._assoc_ops and op_arity == 2 and len(kids) >= 2:
                    out.extend([tok] * (len(kids) - 1))
                    return True
                if len(kids) != op_arity:
                    return False
                out.append(tok)
                return True
            return False

        ok = walk(node)
        return out if ok else None

    def _canonicalize_tokens(self, toks):
        """Project one token sequence into canonical short form when possible."""
        ast = self._rpn_to_ast(toks)
        if ast is None:
            return toks
        can = self._canon(ast)
        can_toks = self._ast_to_rpn(can)
        return can_toks if can_toks else toks

    def _canonicalize_batch_targets(self, seqs: torch.Tensor, last_pos: torch.Tensor, indices: torch.Tensor):
        """Build canonical teacher-forcing targets for selected rows only (e.g., elites)."""
        k = int(indices.shape[0])
        max_len = int(seqs.shape[1])
        canon = torch.zeros((k, max_len), dtype=seqs.dtype, device=self.device)
        canon_len = torch.zeros(k, dtype=last_pos.dtype, device=self.device)
        min_len = int(ModelConfig.MIN_FORMULA_LEN)
        for row, idx in enumerate(indices.tolist()):
            L = int(last_pos[idx].item())
            raw_toks = seqs[idx, :L].tolist()
            can_toks = self._canonicalize_tokens(raw_toks)
            # keep training/generation min-length constraints consistent
            if len(can_toks) < min_len:
                can_toks = raw_toks
            Lc = len(can_toks)
            if Lc > 0:
                canon[row, :Lc] = torch.tensor(can_toks, device=self.device, dtype=torch.long)
            canon_len[row] = Lc
        return canon, canon_len

    def _remember_key(self, key64):
        """Remember a formula key with FIFO eviction to cap memory."""

        if key64 in self._seen_global:
            return
        self._seen_global.add(key64)
        self._seen_fifo.append(key64)
        if len(self._seen_fifo) > self._seen_limit:
            old = self._seen_fifo.popleft()
            self._seen_global.discard(old)

    def _remember_sem_key(self, key64: int) -> None:
        if key64 in self._seen_sem_global:
            return
        self._seen_sem_global.add(key64)
        self._seen_sem_fifo.append(key64)
        if len(self._seen_sem_fifo) > self._seen_sem_limit:
            old = self._seen_sem_fifo.popleft()
            self._seen_sem_global.discard(old)

    def _remember_can_key(self, key64: int) -> None:
        if key64 in self._seen_can_global:
            return
        self._seen_can_global.add(key64)
        self._seen_can_fifo.append(key64)
        if len(self._seen_can_fifo) > self._seen_can_limit:
            old = self._seen_can_fifo.popleft()
            self._seen_can_global.discard(old)

    @ torch.no_grad()
    def _sample_one_formula(self):
        """Sample exactly one formula (tokens, length, ops_count) using the same constraints as batch sampling."""
        inp = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        depth = torch.zeros(1, dtype=torch.long, device=self.device)
        ops_count = torch.zeros(1, dtype=torch.long, device=self.device)
        pad_id = 0
        min_len = ModelConfig.MIN_FORMULA_LEN
        stop_eps = ModelConfig.STOP_PROB_EPS
        tokens: list[int] = []
        for t in range(ModelConfig.MAX_FORMULA_LEN):
            mask = self._build_strict_mask_rpn(depth, t)
            logits, _, _, _, _ = self.model(inp)
            dist = CatDist(logits=logits + mask)
            action = dist.sample()  # [1]
            tok = int(action.item())
            tokens.append(tok)
            inp = torch.cat([inp, action.view(1, 1)], dim=1)
            arity = self.arity_vec[action]  # [1]
            is_feat = (arity == 0)
            depth = torch.where(is_feat, depth + 1, depth - arity + 1)
            ops_count = ops_count + (arity > 0).long()
            step_pos = t + 1
            # stop decision only when formula can terminate
            _, _, _, _, stop_logit = self.model(inp)
            can_stop = (depth == 1) & (step_pos >= min_len)
            if bool(can_stop.item()):
                p_stop = torch.sigmoid(stop_logit).clamp(stop_eps, 1.0 - stop_eps)
                do_stop = bool(Bernoulli(probs=p_stop).sample().item())
                if do_stop:
                    return tokens, step_pos, int(ops_count.item())
        return tokens, ModelConfig.MAX_FORMULA_LEN, int(ops_count.item())

    def _dedup_batch_inplace(self, seqs: torch.Tensor, last_pos: torch.Tensor, ops_count: torch.Tensor):
        """In-place de-dup: ensure no exact duplicate formulas in this batch (and optionally globally)."""
        seen_batch_raw = set()
        seen_batch_can = set()
        seen_sem_batch = set()
        pad_id = 0
        formulas = [None] * int(seqs.shape[0])
        for i in range(int(seqs.shape[0])):
            L = int(last_pos[i].item())
            raw_toks = seqs[i, :L].tolist()
            toks = self._canonicalize_tokens(raw_toks)
            raw_key = self._key64(raw_toks)
            can_key = self._key64(toks)
            sem_key = self._sem_key64(toks) if self._use_sem_dedup else 0
            dup = (
                    (raw_key in seen_batch_raw)
                    or (raw_key in self._seen_global)
                    or (can_key in seen_batch_can)
                    or (can_key in self._seen_can_global)
            )
            if self._use_sem_dedup:
                dup = dup or (sem_key in seen_sem_batch) or (sem_key in self._seen_sem_global)
            if dup:
            # rejection resample
                for _ in range(self._dedup_max_tries):
                    toks2, L2, ops2 = self._sample_one_formula()
                    can_toks2 = self._canonicalize_tokens(toks2)
                    raw_key2 = self._key64(toks2)
                    can_key2 = self._key64(can_toks2)
                    sem2 = self._sem_key64(can_toks2) if self._use_sem_dedup else 0
                    ok = (
                            (raw_key2 not in seen_batch_raw)
                            and (raw_key2 not in self._seen_global)
                            and (can_key2 not in seen_batch_can)
                            and (can_key2 not in self._seen_can_global)
                    )
                    if self._use_sem_dedup:
                        ok = ok and (sem2 not in seen_sem_batch) and (sem2 not in self._seen_sem_global)
                    if ok:
                        seqs[i].fill_(pad_id)
                        seqs[i, :L2] = torch.tensor(toks2, device=self.device, dtype=torch.long)
                        last_pos[i] = L2
                        ops_count[i] = ops2
                        toks = can_toks2
                        raw_key = raw_key2
                        can_key = can_key2
                        sem_key = sem2
                        break
                    else:
                        self._dedup_fail += 1
            seen_batch_raw.add(raw_key)
            self._remember_key(raw_key)
            seen_batch_can.add(can_key)
            self._remember_can_key(can_key)
            if self._use_sem_dedup:
                seen_sem_batch.add(sem_key)
                self._remember_sem_key(sem_key)
            formulas[i] = toks
        return formulas

    def _build_strict_mask_rpn(self, depth: torch.Tensor, step: int) -> torch.Tensor:
        """
        depth: [B] 当前栈深度（stack size）
        返回: [B, vocab_size] 的 mask（允许=0，禁止=-inf）
        """
        B = depth.shape[0]
        V = self.vocab_size
        device = depth.device
        # [B, V]
        arity = self.arity_vec.unsqueeze(0).expand(B, V)
        depth_b = depth.unsqueeze(1).expand(B, V)
        # 选择该 token 后的栈深度
        depth_after = torch.where(
            arity == 0,  # feature
            depth_b + 1,
            depth_b - arity + 1  # op: -k + 1
        )
        # 条件1：不能 underflow（选 op 时必须 depth >= arity）
        valid_underflow = (arity == 0) | (depth_b >= arity)
        # 条件2：必须“可收敛”：剩余步数内至少存在一种方式能把 depth_after 收敛到 1
        r_after = ModelConfig.MAX_FORMULA_LEN - (step + 1)
        if r_after == 0:
            finishable = (depth_after == 1)
        elif self.max_reduce <= 0:
            # 如果所有 op 都是 unary（arity-1=0），栈永远降不下去：必须一直保持 1
            finishable = (depth_after == 1)
        else:
            # 最乐观情况下，每一步最多把栈深度减少 max_reduce
            # 要从 d 收敛到 1，至少要减少 (d-1)
            finishable = (depth_after <= (self.max_reduce * r_after + 1))
        allowed = valid_underflow & finishable
        mask = torch.full((B, V), float("-inf"), device=device, dtype=torch.float32)
        mask.masked_fill_(allowed, 0.0)
        return mask

    def train(self):
        print("🚀 Starting Meme Alpha Mining with LoRD Regularization..." if self.use_lord else "🚀 Starting Meme Alpha Mining...")
        if self.use_lord:
            print(f"   LoRD Regularization enabled")
            print(f"   Target keywords: ['q_proj', 'k_proj', 'attention', 'qk_norm']")
        
        pbar = tqdm(range(ModelConfig.TRAIN_STEPS))
        elite_frac = float(getattr(ModelConfig, "CEM_ELITE_FRAC", 0.10))
        elite_frac = max(1.0 / ModelConfig.BATCH_SIZE, min(elite_frac, 0.5))
        entropy_beta = float(getattr(ModelConfig, "CEM_ENTROPY_BETA", 0.5))
        thr_bins_list = sorted([float(x) for x in ModelConfig.THRESH_BINS])
        if len(thr_bins_list) >= 3:
            step0 = thr_bins_list[1] - thr_bins_list[0]
        # 允许一点浮点误差
        for j in range(2, len(thr_bins_list)):
            if abs((thr_bins_list[j] - thr_bins_list[j - 1]) - step0) > 1e-9:
                raise ValueError("THRESH_BINS must be uniformly spaced for coarse/refine search.")
        thr_bins = torch.tensor(thr_bins_list, device=self.device, dtype=torch.float32)

        for step in pbar:
            bs = ModelConfig.BATCH_SIZE
            inp = torch.zeros((bs, 1), dtype=torch.long, device=self.device)

            tokens_list = []

            depth = torch.zeros(bs, dtype=torch.long, device=self.device)
            alive = torch.ones(bs, dtype=torch.bool, device=self.device)
            last_pos = torch.zeros(bs, dtype=torch.long, device=self.device)  # in inp positions
            ops_count = torch.zeros(bs, dtype=torch.long, device=self.device)
            pad_id = 0  # placeholder token id (not executed; we slice by length)
            min_len = ModelConfig.MIN_FORMULA_LEN
            lam_ops = ModelConfig.OPS_PENALTY_LAMBDA
            lam_len = ModelConfig.LEN_PENALTY_LAMBDA
            stop_eps = ModelConfig.STOP_PROB_EPS
            alive_total = 0
            can_stop_total = 0
            do_stop_total = 0
            p_stop_sum = 0.0
            p_stop_cnt = 0
            for step_in_formula in range(ModelConfig.MAX_FORMULA_LEN):
                alive_before = alive
                alive_total += int(alive_before.sum().item())
                mask = self._build_strict_mask_rpn(depth, step_in_formula)  # [B, V]
                logits, _, _, _, _ = self.model(inp)
                dist = CatDist(logits=logits + mask)
                sampled = dist.sample()
                action = torch.where(alive_before, sampled, torch.full_like(sampled, pad_id))
                tokens_list.append(action)
                inp = torch.cat([inp, action.unsqueeze(1)], dim=1)
                # 更新栈深度
                arity = self.arity_vec[action]  # [B]
                is_feat = (arity == 0)
                new_depth = torch.where(is_feat, depth + 1, depth - arity + 1)
                depth = torch.where(alive_before, new_depth, depth)
                ops_count = ops_count + (alive_before & (arity > 0)).long()
                # 记录当前真实结束位置（action appended at position step_in_formula+1）
                step_pos = step_in_formula + 1
                last_pos = torch.where(alive_before, torch.full_like(last_pos, step_pos), last_pos)
                # stop decision: only meaningful when formula is already complete (depth==1) and length>=min_len
                # NOTE: stop is predicted from the post-append state.
                _, _, _, _, stop_logit = self.model(inp)
                can_stop = alive_before & (depth == 1) & (step_pos >= min_len)
                p_stop = torch.sigmoid(stop_logit).clamp(stop_eps, 1.0 - stop_eps)
                stop_dist = Bernoulli(probs=p_stop)
                stop_sample = stop_dist.sample()
                stop_sample = torch.where(can_stop, stop_sample, torch.zeros_like(stop_sample))
                do_stop = can_stop & (stop_sample.bool())
                alive = alive_before & (~do_stop)
                if can_stop.any():
                    c = int(can_stop.sum().item())
                    can_stop_total += c
                    do_stop_total += int(do_stop.sum().item())
                    p_stop_sum += float(p_stop[can_stop].sum().item())
                    p_stop_cnt += c
            seqs = torch.stack(tokens_list, dim=1)  # [B, MAX_LEN]
            best_thr_idx = torch.zeros(bs, dtype=torch.long, device=self.device)  # for optional supervised thr-head
            formulas = self._dedup_batch_inplace(seqs, last_pos, ops_count)
            rewards = torch.zeros(bs, device=self.device)
            raw_scores = torch.full((bs,), float("nan"), device=self.device)  # store best raw score (oracle thr)
            invalid = 0
            lowvar = 0
            reason_hist = {}
            bad_examples = []
            for i in range(bs):
                L = int(last_pos[i].item())
                formula = formulas[i]
                res, info   = self.vm.execute(formula, self.loader.feat_tensor)
                # res, info  = self.vm.execute(seqs[i].tolist(), self.loader.feat_tensor)
                if res is None:
                    invalid += 1
                    r = info.get("reason", "unknown")
                    reason_hist[r] = reason_hist.get(r, 0) + 1
                    rewards[i] = -1.0

                    if len(bad_examples) < 3:
                        bad_examples.append({"formula": formula, "info": info})
                    continue

                '''if res.std() < 1e-4:
                    lowvar += 1
                    rewards[i] = -10.0
                    continue'''

                # ---- two-stage threshold search (coarse->refine) ----best_score = None
                thr_min = thr_bins_list[0]
                thr_max = thr_bins_list[-1]
                thr_step = thr_bins_list[1] - thr_bins_list[0] if len(thr_bins_list) > 1 else 1.0
                coarse_step = float(getattr(ModelConfig, "THRESH_COARSE_STEP", 0.05))
                refine_radius = float(getattr(ModelConfig, "THRESH_REFINE_RADIUS", 0.05))

                def thr_to_idx(thr: float) -> int:
                    return int(round((thr - thr_min) / thr_step))
                coarse_thrs = []
                x = thr_min
                while x <= thr_max + 1e-12:
                    coarse_thrs.append(x)
                    x += coarse_step
                coarse_idx = [thr_to_idx(t) for t in coarse_thrs if (thr_min - 1e-12) <= t <= (thr_max + 1e-12)]
                coarse_idx = [k for k in coarse_idx if 0 <= k < len(thr_bins_list)]
                coarse_idx = sorted(set(coarse_idx))
                best_score = None
                best_k = coarse_idx[0] if coarse_idx else 0
                best_details = None
                best_ret = None
                for k in coarse_idx:
                    thr = thr_bins_list[k]
                    score_k, ret_k, details_k = self.bt.evaluate(res, self.loader.raw_data_cache,
                                                                 self.loader.target_ret, float(thr))
                    s = float(score_k.item())
                    if (best_score is None) or (s > best_score):
                        best_score = s
                        best_k = k
                        best_details = details_k
                        best_ret = ret_k
                radius_steps = int(round(refine_radius / thr_step))
                lo = max(0, best_k - radius_steps)
                hi = min(len(thr_bins_list) - 1, best_k + radius_steps)
                for k in range(lo, hi + 1):
                    thr = thr_bins_list[k]
                    score_k, ret_k, details_k = self.bt.evaluate(res, self.loader.raw_data_cache,
                                                                 self.loader.target_ret, float(thr))
                    s = float(score_k.item())
                    if s > best_score:
                        best_score = s
                        best_k = k
                        best_details = details_k
                        best_ret = ret_k
                score = torch.tensor(best_score, device=self.device, dtype=torch.float32)
                best_thr_idx[i] = int(best_k)
                raw_scores[i] = score
                ret_val, details = best_ret, best_details
                # length penalty: reward -= λ_ops * (#ops) + λ_len * (raw token length)
                penalty = lam_ops * float(ops_count[i].item()) + lam_len * float(last_pos[i].item())
                reward_i = float(score.item()) - penalty
                rewards[i] = torch.tensor(reward_i, device=self.device, dtype=torch.float32)

                if reward_i > self.best_score:
                    self.best_score = reward_i
                    self.best_raw_score = float(score.item())
                    self.best_formula = formula
                    self.best_threshold = float(thr_bins[best_thr_idx[i]].item())
                    tqdm.write(
                        f"[!] New King: Reward {reward_i:.6f} | RawScore {float(score.item()):.6f} | "
                        f"Ret {ret_val:.2%} | Formula {formula}"
                    )
                    print(details)
            early_stop_ratio = (last_pos < ModelConfig.MAX_FORMULA_LEN).float().mean().item()
            stop_able_ratio = (can_stop_total / max(alive_total, 1))
            stop_hit_ratio = (do_stop_total / max(can_stop_total, 1))
            stop_p_mean = (p_stop_sum / max(p_stop_cnt, 1))
            valid_raw = raw_scores[~torch.isnan(raw_scores)]
            raw_score_mean = valid_raw.mean().item() if valid_raw.numel() > 0 else float("nan")
            if bad_examples:
                tqdm.write(f"BadExamples={bad_examples}")
            # ---- CEM: select elites ----
            elite_k = max(1, int(bs * elite_frac))
            elite_idx = torch.topk(rewards, k=elite_k, largest=True).indices  # [K]
            # ---- CEM: behavior cloning on elites (maximize likelihood) ----
            # Teacher forcing is done on canonicalized elite sequences so the model learns
            # to output normalized (often shorter) formulas directly.
            cem_loss = torch.tensor(0.0, device=self.device)
            ent_loss = torch.tensor(0.0, device=self.device)
            stop_loss = torch.tensor(0.0, device=self.device)
            stop_coef = float(getattr(ModelConfig, "STOP_LOSS_COEF", 1.0))
            seqs_e, elite_can_last_pos = self._canonicalize_batch_targets(seqs, last_pos, elite_idx)
            # teacher forcing: step-by-step on canonical prefixes
            inp_e = torch.zeros((elite_k, 1), dtype=torch.long, device=self.device)
            depth_e = torch.zeros(elite_k, dtype=torch.long, device=self.device)
            logits_e, _, _, _, _ = self.model(inp_e)
            for t in range(ModelConfig.MAX_FORMULA_LEN):
                mask_e = self._build_strict_mask_rpn(depth_e, t)  # [K, V]
                masked_logits = logits_e + mask_e
                # canonical targets at step t
                y = seqs_e[:, t]  # [K]
                # only positions <= canonical length are real tokens
                valid = (t + 1 <= elite_can_last_pos)
                if valid.any():
                    ce = F.cross_entropy(masked_logits[valid], y[valid], reduction="mean")
                    cem_loss = cem_loss + ce
                # entropy bonus (encourage exploration / prevent collapse)
                if entropy_beta > 0 and valid.any():
                    dist_e = CatDist(logits=masked_logits[valid])
                    ent_loss = ent_loss - entropy_beta * dist_e.entropy().mean()
                # update depth using canonical token y on valid positions only
                ar = self.arity_vec[y]
                is_feat = (ar == 0)
                new_depth_e = torch.where(is_feat, depth_e + 1, depth_e - ar + 1)
                depth_post = torch.where(valid, new_depth_e, depth_e)
                # stop supervision must use post-append state (same as sampling loop)
                inp_post = torch.cat([inp_e, y.unsqueeze(1)], dim=1)
                logits_next, _, _, _, stop_logit_post = self.model(inp_post)
                can_stop_post = valid & (depth_post == 1) & (t + 1 >= min_len)
                stop_target = (elite_can_last_pos == (t + 1)) & can_stop_post
                if can_stop_post.any():
                    s_loss = F.binary_cross_entropy_with_logits(
                        stop_logit_post[can_stop_post],
                        stop_target[can_stop_post].float(),
                        reduction="mean",
                    )
                    stop_loss = stop_loss + s_loss
                depth_e = depth_post
                inp_e = inp_post
                logits_e = logits_next
            # optional: supervised threshold head on elites (using oracle best_thr_idx)
            _, _, _, thr_logits_e, _ = self.model(inp_e, last_positions=elite_can_last_pos)
            thr_loss = F.cross_entropy(thr_logits_e, best_thr_idx[elite_idx], reduction="mean")
            loss = cem_loss + thr_loss + ent_loss + stop_coef * stop_loss

            tqdm.write(
                f"with seed{self.seed}, InvalidRatio={invalid / bs:.2%} | LowVarRatio={lowvar / bs:.2%} | "
                f"LenMean={last_pos.float().mean().item():.2f} | "
                f"OpsMean={ops_count.float().mean().item():.2f} | EarlyStop={early_stop_ratio:.2%} | "
                f"StopAbleRatio={stop_able_ratio:.2%} | StopHitRatio={stop_hit_ratio:.2%} | StopPMean={stop_p_mean:.4f} | "
                f"RawScoreMean={raw_score_mean:.3f} | "
                f"step={step} cem_loss={float(cem_loss.item()):.3f} thr_loss={float(thr_loss.item()):.3f} "
                f"stop_loss={float(stop_loss.item()):.3f}"
            )
            
            # Gradient step
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            
            # Apply Low-Rank Decay regularization
            if self.use_lord:
                self.lord_opt.step()
            
            # Logging
            avg_reward = rewards.mean().item()
            postfix_dict = {
                'AvgRew': f"{avg_reward:.3f}",
                'BestScore': f"{self.best_score:.3f}",
                'BestRaw': f"{self.best_raw_score:.3f}",
                'CEM': f"{float(cem_loss.item()):.3f}",
                'THR': f"{float(thr_loss.item()):.3f}",
                'STOP': f"{float(stop_loss.item()):.3f}",
            }
            
            if self.use_lord and step % 100 == 0:
                stable_rank = self.rank_monitor.compute()
                postfix_dict['Rank'] = f"{stable_rank:.2f}"
                self.training_history['stable_rank'].append(stable_rank)
            
            self.training_history['step'].append(step)
            self.training_history['avg_reward'].append(avg_reward)
            self.training_history['best_score'].append(self.best_score)
            
            pbar.set_postfix(postfix_dict)

        # Save best formula
        os.makedirs(self.output_dir, exist_ok=True)
        strategy_file = os.path.join(self.output_dir, f"best_meme_strategy_seed{self.seed}.json")
        with open(strategy_file, "w") as f:
            json.dump({"formula": self.best_formula, "threshold": self.best_threshold,
                       "best_reward": self.best_score,
                       "best_raw_score": self.best_raw_score,
                       "threshold_bins": ModelConfig.THRESH_BINS}, f, indent = 2)
        
        # Save training history
        import json as js
        history_file = os.path.join(self.output_dir, f"training_history_seed{self.seed}.json")
        with open(history_file, "w") as f:
            js.dump(self.training_history, f)
        
        print(f"\n✓ Training completed!")
        print(f"  Seed: {self.seed} | Device: {self.device}")
        print(f"  Best reward: {self.best_score:.4f} | Raw score: {self.best_raw_score:.4f}")
        print(f"  Best formula: {self.best_formula}")
        print(f"  Saved strategy to: {strategy_file}")
        print(f"  Saved history to: {history_file}")


def train_one_seed(seed, *, device_str, use_lord, output_dir):
    """Run one seed on the assigned device and return best score."""
    print(f"[SeedRunner] seed={seed} device={device_str} -> start")
    engine = AlphaEngine(seed=seed, device=device_str, use_lord_regularization=use_lord, output_dir=output_dir)
    engine.train()
    best = float(engine.best_score)
    print(f"[SeedRunner] seed={seed} device={device_str} -> done, best={best:.6f}")
    return best


def gpu_worker(gpu_id, seed_queue, results_dict, failures_dict, *, use_lord, output_dir):
    """Consume seeds from queue and train them on one GPU."""
    torch.set_num_threads(1)
    device_str = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(gpu_id)
        except Exception as e:
            print(f"[Worker-{gpu_id}] warning: set_device failed: {e}")
        try:
            torch.set_float32_matmul_precision('medium')
        except Exception:
            pass
    print(f"[Worker-{gpu_id}] started pid={os.getpid()} device={device_str}")

    while True:
        try:
            seed = seed_queue.get_nowait()
        except _q.Empty:
            break

        try:
            best = train_one_seed(seed, device_str=device_str, use_lord=use_lord, output_dir=output_dir)
            results_dict[seed] = best
        except Exception as exc:
            failures_dict[seed] = str(exc)
            print(f"[Worker-{gpu_id}] seed={seed} failed: {exc}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def launch_multi_gpu_seed_jobs(gpu_ids, seeds, workers_per_gpu, use_lord, output_dir):
    """Parallel seed scheduler: queue-based multi-process workers over GPUs."""
    if not seeds:
        raise ValueError("SEEDS 不能为空")
    if workers_per_gpu < 1:
        raise ValueError(f"workers_per_gpu 必须 >= 1，当前={workers_per_gpu}")

    if torch.cuda.is_available():
        visible_gpus = list(gpu_ids)
    else:
        visible_gpus = [0]
        print("[Scheduler] CUDA not available, fallback to CPU workers")

    seed_queue = mp.Queue()
    for seed in seeds:
        seed_queue.put(seed)

    manager = mp.Manager()
    results = manager.dict()
    failures = manager.dict()
    procs = []

    for gid in visible_gpus:
        for _ in range(workers_per_gpu):
            p = mp.Process(
                target=gpu_worker,
                args=(gid, seed_queue, results, failures),
                kwargs={"use_lord": use_lord, "output_dir": output_dir},
            )
            p.start()
            procs.append(p)

    for p in procs:
        p.join()

    results = {int(k): float(v) for k, v in results.items()}
    failures = {int(k): str(v) for k, v in failures.items()}

    print("\n===== Cross-Seed Summary =====")
    out = []
    for seed in seeds:
        if seed in results:
            val = results[seed]
            out.append((seed, val))
            print(f"seed={seed} best={val:.6f}")
        else:
            err = failures.get(seed, "unknown error")
            print(f"seed={seed} failed={err}")

    valid_vals = [v for _, v in out if np.isfinite(v)]
    if out:
        best_seed, best_val = max(out, key=lambda x: x[1] if np.isfinite(x[1]) else -np.inf)
        print(f"[BEST] seed={best_seed} best_score={best_val:.6f}")
    if valid_vals:
        print(f"[AVG] across {len(valid_vals)} seeds mean(best_score)={float(np.mean(valid_vals)):.6f}")

    if failures:
        raise RuntimeError(f"{len(failures)} seed(s) failed: {failures}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    launch_multi_gpu_seed_jobs(
        gpu_ids=ModelConfig.GPUS,
        seeds=ModelConfig.SEEDS,
        workers_per_gpu=ModelConfig.SEEDS_PER_GPU,
        use_lord=ModelConfig.USE_LORD,
        output_dir=ModelConfig.OUTPUT_DIR,
    )