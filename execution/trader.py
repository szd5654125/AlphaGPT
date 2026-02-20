import asyncio
from loguru import logger
from solders.pubkey import Pubkey
from solana.rpc.types import TokenAccountOpts
from config.general_config import ExecutionConfig
from execution.rpc_handler import QuickNodeClient
from execution.jupiter import JupiterAggregator

class SolanaTrader:
    def __init__(self):
        self.rpc = QuickNodeClient()
        self.jup = JupiterAggregator()
        self.is_running = True
        self.TOKEN_PROGRAM_ID = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")

    async def buy(self, token_address: str, amount_sol: float, slippage_bps=500):
        logger.info(f"Executing BUY: {amount_sol} SOL -> {token_address}")
        balance = await self.rpc.get_balance()
        if balance < amount_sol + 0.02:
            logger.warning(f"Insufficient SOL balance: {balance}. Needed: {amount_sol + 0.02}")
            return False
        amount_lamports = int(amount_sol * 1e9)
        quote = await self.jup.get_quote(
            input_mint=ExecutionConfig.SOL_MINT,
            output_mint=token_address,
            amount_integer=amount_lamports,
            slippage_bps=slippage_bps
        )
        if not quote:
            logger.error("No quote found.")
            return False
        out_amount = int(quote['outAmount'])
        logger.info(f"Quote received. Est. Output: {out_amount} raw units.")
        b64_tx = await self.jup.get_swap_tx(quote)
        if not b64_tx:
            return False
        try:
            txn = self.jup.deserialize_and_sign(b64_tx)
        except Exception as e:
            logger.error(f"Signing failed: {e}")
            return False
        sig = await self.rpc.send_and_confirm(txn)
        if sig:
            logger.success(f"BUY Successful: {token_address} | Tx: {sig}")
            return True
        return False

    async def sell(self, token_address: str, percentage: float = 1.0, slippage_bps=500):
        logger.info(f"Executing SELL: {percentage*100}% of {token_address} -> SOL")
        raw_balance = 0
        try:
            wallet_pubkey = Pubkey.from_string(ExecutionConfig.WALLET_ADDRESS)
            mint_pubkey = Pubkey.from_string(token_address)
            opts = TokenAccountOpts(
                program_id=self.TOKEN_PROGRAM_ID,
                mint=mint_pubkey
            )
            resp = await self.rpc.client.get_token_accounts_by_owner_json_parsed(
                wallet_pubkey,
                opts
            )
            if resp.value:
                for account_info in resp.value:
                    amount_str = account_info.account.data.parsed['info']['tokenAmount']['amount']
                    raw_balance += int(amount_str)
            logger.info(f"Token Balance Found: {raw_balance} raw units")
            if raw_balance == 0:
                logger.warning(f"No balance found for {token_address}, skipping sell.")
                return False
            sell_amount = int(raw_balance * percentage)
            if sell_amount == 0:
                logger.warning("Sell amount is 0 (too small percentage?)")
                return False
        except Exception as e:
            logger.error(f"Failed to fetch token balance: {e}")
            return False
        quote = await self.jup.get_quote(
            input_mint=token_address,
            output_mint=ExecutionConfig.SOL_MINT,
            amount_integer=sell_amount,
            slippage_bps=slippage_bps
        )
        if not quote:
            logger.error("Sell quote not found.")
            return False
        b64_tx = await self.jup.get_swap_tx(quote)
        if not b64_tx:
            return False
        try:
            txn = self.jup.deserialize_and_sign(b64_tx)
            sig = await self.rpc.send_and_confirm(txn)
            if sig:
                logger.success(f"SELL Successful: {token_address} | Tx: {sig}")
                return True
        except Exception as e:
            logger.error(f"Sell execution failed: {e}")
        return False

    async def close(self):
        await self.rpc.close()
        await self.jup.close()

if __name__ == "__main__": # test
    async def test_run():
        trader = SolanaTrader()
        BONK_ADDRESS = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"
        logger.info("Testing Token Balance Fetch...")
        await trader.sell(BONK_ADDRESS, percentage=0.5)
        await trader.close()
    try:
        asyncio.run(test_run())
    except KeyboardInterrupt:
        pass