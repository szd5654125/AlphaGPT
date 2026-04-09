import torch

class MemeBacktest:
    def __init__(self):
        self.trade_size = 1000.0
        self.min_liq = 500000.0
        self.base_fee = 0.0060

    def evaluate(self, factors, raw_data, target_ret, threshold,
                 tradeable_mask=None, segment_ids=None, n_segments=0):
        """
        评估交易策略。

        旧模式 (tradeable_mask=None):
            factors/target_ret 形状 [N, T]，按 symbol 聚合，取 median。

        新模式 (tradeable_mask 非 None):
            factors/target_ret 形状 [1, T_total]，串联多个 segment。
            - tradeable_mask [1, T_total]: 仅在 True 的位置允许持仓
            - segment_ids [T_total]: 每个时间步所属 segment 索引
            - n_segments: segment 总数
            按 segment 聚合 PnL 后取 median。
        """
        signal = torch.sigmoid(factors)
        position = (signal > float(threshold)).float()

        # 新模式: 用 tradeable_mask 强制非交易区域的仓位为 0
        if tradeable_mask is not None:
            position = position * tradeable_mask.float()

        total_slippage_one_way = self.base_fee
        prev_pos = torch.roll(position, 1, dims=1)
        prev_pos[:, 0] = 0

        turnover = torch.abs(position - prev_pos)
        tx_cost = turnover * total_slippage_one_way
        gross_pnl = position * target_ret
        net_pnl = gross_pnl - tx_cost

        # ---- 聚合 ----
        if segment_ids is not None and n_segments > 0:
            # 使用 scatter_add 按 segment 聚合（全向量化，无 Python 循环）
            seg_returns = torch.zeros(n_segments, device=factors.device)
            seg_returns.scatter_add_(0, segment_ids, net_pnl[0])

            seg_activity = torch.zeros(n_segments, device=factors.device)
            seg_activity.scatter_add_(0, segment_ids, position[0])

            final_fitness = torch.median(seg_returns)
            details = {
                "activity_median": torch.median(seg_activity).item(),
                "turnover_mean": turnover.mean().item(),
                "turnover_sum_mean": float(seg_returns.abs().sum().item()),
            }
            return final_fitness, seg_returns.mean().item(), details
        else:
            # 旧模式：按 symbol（dim=0）聚合
            cum_ret = net_pnl.sum(dim=1)
            score = cum_ret
            activity = position.sum(dim=1)
            final_fitness = torch.median(score)
            details = {
                "activity_median": torch.median(activity).item(),
                "turnover_mean": turnover.mean().item(),
                "turnover_sum_mean": turnover.sum(dim=1).float().mean().item(),
            }
            return final_fitness, cum_ret.mean().item(), details
