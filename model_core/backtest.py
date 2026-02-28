import torch

class MemeBacktest:
    def __init__(self):
        self.trade_size = 1000.0
        self.min_liq = 500000.0
        self.base_fee = 0.0060

    def evaluate(self, factors, raw_data, target_ret, threshold):
        signal = torch.sigmoid(factors)
        position = (signal > float(threshold)).float()
        total_slippage_one_way = self.base_fee
        prev_pos = torch.roll(position, 1, dims=1)
        prev_pos[:, 0] = 0
        turnover = torch.abs(position - prev_pos)
        tx_cost = turnover * total_slippage_one_way
        gross_pnl = position * target_ret
        net_pnl = gross_pnl - tx_cost
        cum_ret = net_pnl.sum(dim=1)
        big_drawdowns = (net_pnl < -0.05).float().sum(dim=1)
        score = cum_ret - (big_drawdowns * 2.0)
        activity = position.sum(dim=1)
        score = torch.where(activity < 5, torch.tensor(-10.0, device=score.device), score)
        final_fitness = torch.median(score)
        details = {"activity_median": torch.median(activity).item(), "turnover_mean": turnover.mean().item(),
                   "turnover_sum_mean": turnover.sum(dim=1).float().mean().item()}
        return final_fitness, cum_ret.mean().item(), details