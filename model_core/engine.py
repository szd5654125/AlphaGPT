import torch
import random
from torch.distributions import Categorical, Bernoulli
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
            inp = torch.zeros((bs, 1), dtype=torch.long, device=self.device)
            
            log_probs_tokens = []
            log_probs_stop = []
            tokens_list = []

            depth = torch.zeros(bs, dtype=torch.long, device=self.device)
            alive = torch.ones(bs, dtype=torch.bool, device=self.device)
            last_pos = torch.zeros(bs, dtype=torch.long, device=self.device)  # in inp positions
            ops_count = torch.zeros(bs, dtype=torch.long, device=self.device)
            pad_id = 0  # placeholder token id (not executed; we slice by length)
            min_len = ModelConfig.MIN_FORMULA_LEN
            lam_ops = ModelConfig.OPS_PENALTY_LAMBDA
            stop_eps = ModelConfig.STOP_PROB_EPS
            logits, _, _, _, _ = self.model(inp)  # current state logits (for first step)
            for step_in_formula in range(ModelConfig.MAX_FORMULA_LEN):
                alive_before = alive
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
                # NOTE: stop is predicted from the post-append state.
                logits, _, _, _, stop_logit = self.model(inp)
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
            _, value, _, thr_logits, _ = self.model(inp, last_positions=last_pos)  # [B, n_thr]
            value = value.squeeze(-1)  # -> [B]

            thr_dist = Categorical(logits=thr_logits)
            thr_idx = thr_dist.sample()  # [B]
            logp_thr = thr_dist.log_prob(thr_idx)  # [B]
            thr_bins = torch.tensor(ModelConfig.THRESH_BINS, device=self.device,
                                                 dtype=torch.float32)
            thr_val = thr_bins[thr_idx]  # [B]
            
            rewards = torch.zeros(bs, device=self.device)
            raw_scores = torch.full((bs,), float("nan"), device=self.device)
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
                raw_scores[i] = score
                # length penalty: reward -= λ * (#ops)
                penalty = lam_ops * float(ops_count[i].item())
                reward_i = score - torch.tensor(penalty, device=rewards.device, dtype=rewards.dtype)
                rewards[i] = reward_i

                if reward_i.item() > self.best_score:
                    self.best_score = reward_i.item()
                    self.best_raw_score = score.item()
                    self.best_formula = formula
                    self.best_threshold = float(thr_val[i].item())
                    tqdm.write(
                        f"[!] New King: Reward {reward_i:.2f} | RawScore {score:.2f} | "
                        f"Ret {ret_val:.2%} | Formula {formula}"
                    )
                    print(details)
            early_stop_ratio = (last_pos < ModelConfig.MAX_FORMULA_LEN).float().mean().item()
            stop_able_ratio = torch.stack(log_probs_stop, dim=1).ne(0).any(dim=1).float().mean().item()
            stop_p_mean = torch.sigmoid(stop_logit).mean().item()
            valid_raw = raw_scores[~torch.isnan(raw_scores)]
            raw_score_mean = valid_raw.mean().item() if valid_raw.numel() > 0 else float("nan")
            if bad_examples:
                tqdm.write(f"BadExamples={bad_examples}")
            # --- Actor-Critic advantage with baseline ---
            adv = rewards - value.detach()
            adv = (adv - adv.mean()) / (adv.std() + ModelConfig.ADV_NORM_EPS)
            
            logp_tokens = torch.stack(log_probs_tokens, dim=1).sum(dim=1)  # [B]
            logp_stop_total = torch.stack(log_probs_stop, dim=1).sum(dim=1)  # [B]
            logp_total = logp_tokens + logp_stop_total + logp_thr  # [B]
            # policy loss
            policy_loss = -(logp_total * adv).mean()
            # critic loss（Huber 更抗 reward outlier；也可换 MSE）
            value_loss = F.smooth_l1_loss(value, rewards)
            loss = policy_loss + ModelConfig.VALUE_LOSS_COEF * value_loss

            tqdm.write(
                f"with seed{self.seed}, InvalidRatio={invalid / bs:.2%} | LowVarRatio={lowvar / bs:.2%} | "
                f"ThrMean={thr_val.mean().item():.3f} | LenMean={last_pos.float().mean().item():.2f} | "
                f"OpsMean={ops_count.float().mean().item():.2f} | EarlyStop={early_stop_ratio:.2%} | "
                f"StopAbleRatio={stop_able_ratio:.2%} | StopPMean={stop_p_mean:.4f} | "
                f"RawScoreMean={raw_score_mean:.3f} | "
                f"step={step} ploss={policy_loss.item():.3f} vloss={value_loss.item():.3f}"
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
                'VLoss': f"{value_loss.item():.3f}",
                'PLoss': f"{policy_loss.item():.3f}",
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