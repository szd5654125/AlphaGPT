from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed
from loguru import logger
from execution.config import ExecutionConfig

class QuickNodeClient:
    def __init__(self):
        self.client = AsyncClient(ExecutionConfig.RPC_URL, commitment=Confirmed)

    async def get_balance(self):
        try:
            resp = await self.client.get_balance(ExecutionConfig.PAYER_KEYPAIR.pubkey())
            return resp.value / 1e9
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return 0.0

    async def get_token_balance(self, mint_address: str):
        pass

    async def send_and_confirm(self, txn, max_retries=3):
        try:
            signature = await self.client.send_transaction(
                txn,
                opts=None
            )
            logger.info(f"Transaction Sent: {signature.value}")
            sig_str = str(signature.value)
            await self.client.confirm_transaction(signature.value)
            logger.success(f"Transaction Confirmed: https://solscan.io/tx/{sig_str}")
            return sig_str
        except Exception as e:
            logger.error(f"Transaction Failed: {e}")
            return None

    async def close(self):
        await self.client.close()