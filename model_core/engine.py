import torch
import random
from torch.distributions import Categorical, Bernoulli
from tqdm import tqdm
import json
from model_core.factors import FeatureEngineer
from model_core.ops import OPS_CONFIG
from config.general_config import ModelConfig
from model_core.data_loader_csv import CsvCryptoDataLoader, CsvLoaderConfig
from model_core.alphagpt import AlphaGPT, NewtonSchulzLowRankDecay, StableRankMonitor
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest


class AlphaEngine:
    def __init__(self, use_lord_regularization=True, lord_decay_rate=1e-3, lord_num_iterations=5):
        """
        Initialize AlphaGPT training engine.
        
        Args:
            use_lord_regularization: Enable Low-Rank Decay (LoRD) regularization
            lord_decay_rate: Strength of LoRD regularization
            lord_num_iterations: Number of Newton-Schulz iterations per step
        """
        random.seed(ModelConfig.SEED)
        torch.manual_seed(ModelConfig.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(ModelConfig.SEED)
            torch.cuda.manual_seed_all(ModelConfig.SEED)
        '''self.loader = CryptoDataLoader()
        self.loader.load_data()'''
        cfg = CsvLoaderConfig(
            csv_paths=["data/futures_um_monthly_klines_ETHUSDT_5m_0_53.csv"],  # 改成你的路径
            device=ModelConfig.DEVICE,
            max_symbols=50,  # 你可以先小一点试跑
            liquidity_mode="quote_volume",
        )
        # cuda_snapshot("engine::__init__::start", ModelConfig.DEVICE)
        self.loader = CsvCryptoDataLoader(cfg).load_data()
        # cuda_snapshot("engine::__init__::after_load_data", ModelConfig.DEVICE)
        self.model = AlphaGPT().to(ModelConfig.DEVICE)
        # cuda_snapshot("engine::__init__::after_model_to", ModelConfig.DEVICE)
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
        self.arity_vec = torch.zeros(self.vocab_size, dtype=torch.long, device=ModelConfig.DEVICE)
        for j, (_, _, arity) in enumerate(OPS_CONFIG):
            self.arity_vec[self.feat_offset + j] = arity
        # 单步最大“降栈幅度”：max(arity-1)，用于判断“剩余步数是否还能收敛到 1”
        self.max_reduce = max((arity - 1 for (_, _, arity) in OPS_CONFIG), default=0)

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
        
        for step in pbar:
            bs = ModelConfig.BATCH_SIZE
            inp = torch.zeros((bs, 1), dtype=torch.long, device=ModelConfig.DEVICE)
            
            log_probs_tokens = []
            log_probs_stop = []
            tokens_list = []

            depth = torch.zeros(bs, dtype=torch.long, device=ModelConfig.DEVICE)
            alive = torch.ones(bs, dtype=torch.bool, device=ModelConfig.DEVICE)
            last_pos = torch.zeros(bs, dtype=torch.long, device=ModelConfig.DEVICE)  # in inp positions
            ops_count = torch.zeros(bs, dtype=torch.long, device=ModelConfig.DEVICE)
            pad_id = 0  # placeholder token id (not executed; we slice by length)
            min_len = ModelConfig.MIN_FORMULA_LEN
            lam_ops = ModelConfig.OPS_PENALTY_LAMBDA
            stop_eps = ModelConfig.STOP_PROB_EPS
            for step_in_formula in range(ModelConfig.MAX_FORMULA_LEN):
                # cuda_snapshot("train::before_first_forward", ModelConfig.DEVICE, extra=f"inp={tuple(inp.shape)}")
                alive_before = alive
                logits, _, _, _, stop_logit = self.model(inp)  # [B,V], [B]
                # cuda_snapshot("train::after_first_forward", ModelConfig.DEVICE, extra=f"logits={tuple(logits.shape)}")
                mask = self._build_strict_mask_rpn(depth, step_in_formula)  # [B, V]
                dist = Categorical(logits=logits + mask)
                sampled = dist.sample()
                action = torch.where(alive_before, sampled, torch.full_like(sampled, pad_id))
                lp_tok = dist.log_prob(sampled) * alive_before.float()
                log_probs_tokens.append(lp_tok)
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
                can_stop = alive_before & (depth == 1) & (step_pos >= min_len)
                p_stop = torch.sigmoid(stop_logit).clamp(stop_eps, 1.0 - stop_eps)
                stop_dist = Bernoulli(probs=p_stop)
                stop_sample = stop_dist.sample()
                stop_sample = torch.where(can_stop, stop_sample, torch.zeros_like(stop_sample))
                lp_stop = stop_dist.log_prob(stop_sample) * can_stop.float()
                log_probs_stop.append(lp_stop)
                do_stop = can_stop & (stop_sample.bool())
                alive = alive_before & (~do_stop)
            
            seqs = torch.stack(tokens_list, dim=1)  # [B, MAX_LEN]
            _, _, _, thr_logits, _ = self.model(inp, last_positions=last_pos)  # [B, n_thr]
            thr_dist = Categorical(logits=thr_logits)
            thr_idx = thr_dist.sample()  # [B]
            logp_thr = thr_dist.log_prob(thr_idx)  # [B]
            thr_bins = torch.tensor(ModelConfig.THRESH_BINS, device=ModelConfig.DEVICE,
                                                 dtype=torch.float32)
            thr_val = thr_bins[thr_idx]  # [B]
            
            rewards = torch.zeros(bs, device=ModelConfig.DEVICE)
            invalid = 0
            lowvar = 0
            reason_hist = {}
            bad_examples = []
            for i in range(bs):
                L = int(last_pos[i].item())
                formula = seqs[i, :L].tolist()
                res, info   = self.vm.execute(formula, self.loader.feat_tensor)
                # res, info  = self.vm.execute(seqs[i].tolist(), self.loader.feat_tensor)
                if res is None:
                    invalid += 1
                    r = info.get("reason", "unknown")
                    reason_hist[r] = reason_hist.get(r, 0) + 1
                    rewards[i] = -5.0

                    if len(bad_examples) < 3:
                        bad_examples.append({"formula": formula, "info": info})
                    continue

                if res.std() < 1e-4:
                    lowvar += 1
                    rewards[i] = -10.0
                    continue

                score, ret_val, details = self.bt.evaluate(
                    res, self.loader.raw_data_cache, self.loader.target_ret, thr_val[i].item()
                )
                # length penalty: reward -= λ * (#ops)
                penalty = lam_ops * float(ops_count[i].item())
                rewards[i] = score - torch.tensor(penalty, device=rewards.device, dtype=rewards.dtype)

                if score.item() > self.best_score:
                    self.best_score = score.item()
                    self.best_formula = formula
                    self.best_threshold = float(thr_val[i].item())
                    tqdm.write(f"[!] New King: Score {score:.2f} | Ret {ret_val:.2%} | Formula {formula}")
                    print(details)
            early_stop_ratio = (last_pos < ModelConfig.MAX_FORMULA_LEN).float().mean().item()
            tqdm.write(
                f"InvalidRatio={invalid / bs:.2%} | LowVarRatio={lowvar / bs:.2%} | "
                f"ThrMean={thr_val.mean().item():.3f} | LenMean={last_pos.float().mean().item():.2f} | "
                f"OpsMean={ops_count.float().mean().item():.2f} | EarlyStop={early_stop_ratio:.2%}"
            )
            if bad_examples:
                tqdm.write(f"BadExamples={bad_examples}")
            # Normalize rewards
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            
            logp_tokens = torch.stack(log_probs_tokens, dim=1).sum(dim=1)  # [B]
            logp_stop_total = torch.stack(log_probs_stop, dim=1).sum(dim=1)  # [B]
            logp_total = logp_tokens + logp_stop_total + logp_thr  # [B]
            loss = -(logp_total * adv).mean()
            
            # Gradient step
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            
            # Apply Low-Rank Decay regularization
            if self.use_lord:
                self.lord_opt.step()
            
            # Logging
            avg_reward = rewards.mean().item()
            postfix_dict = {'AvgRew': f"{avg_reward:.3f}", 'BestScore': f"{self.best_score:.3f}"}
            
            if self.use_lord and step % 100 == 0:
                stable_rank = self.rank_monitor.compute()
                postfix_dict['Rank'] = f"{stable_rank:.2f}"
                self.training_history['stable_rank'].append(stable_rank)
            
            self.training_history['step'].append(step)
            self.training_history['avg_reward'].append(avg_reward)
            self.training_history['best_score'].append(self.best_score)
            
            pbar.set_postfix(postfix_dict)

        # Save best formula
        with open("best_meme_strategy.json", "w") as f:
            json.dump({"formula": self.best_formula, "threshold": self.best_threshold,
                       "threshold_bins": ModelConfig.THRESH_BINS}, f, indent = 2)
        
        # Save training history
        import json as js
        with open("training_history.json", "w") as f:
            js.dump(self.training_history, f)
        
        print(f"\n✓ Training completed!")
        print(f"  Best score: {self.best_score:.4f}")
        print(f"  Best formula: {self.best_formula}")


if __name__ == "__main__":
    eng = AlphaEngine(use_lord_regularization=True)
    eng.train()