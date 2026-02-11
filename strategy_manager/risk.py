from strategy_manager.config import StrategyConfig
from execution.jupiter import JupiterAggregator # 复用 Jupiter 做模拟
from loguru import logger

class RiskEngine:
    def __init__(self):
        self.config = StrategyConfig()
        self.jup = JupiterAggregator()

    async def check_safety(self, token_address, liquidity_usd):
        if liquidity_usd < 5000:
            logger.warning(f"[x] Risk: Liquidity too low (${liquidity_usd})")
            return False

        try:
            quote = await self.jup.get_quote(
                input_mint=token_address,
                output_mint="So11111111111111111111111111111111111111112",
                amount_integer=1000000,
                slippage_bps=1000
            )
            if not quote:
                logger.warning(f"[x] Risk: Cannot verify sell path (Honeypot?)")
                return False
        except Exception:
            return False
            
        return True

    def calculate_position_size(self, wallet_balance_sol):
        size = self.config.ENTRY_AMOUNT_SOL
        
        if wallet_balance_sol < size + 0.1:
            return 0.0
            
        return size

    async def close(self):
        await self.jup.close()