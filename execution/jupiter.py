import aiohttp
import base64
from loguru import logger
from solders.transaction import VersionedTransaction
from config.general_config import ExecutionConfig

class JupiterAggregator:
    def __init__(self):
        self.base_url = "https://quote-api.jup.ag/v6"
        self.session = None

    async def _get_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session

    async def get_quote(self, input_mint, output_mint, amount_integer, slippage_bps=None):
        session = await self._get_session()
        slippage = slippage_bps if slippage_bps else ExecutionConfig.DEFAULT_SLIPPAGE_BPS
        url = f"{self.base_url}/quote"
        params = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": str(amount_integer),
            "slippageBps": str(slippage),
            "onlyDirectRoutes": "false",
            "asLegacyTransaction": "false"
        }
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                text = await resp.text()
                logger.error(f"Jupiter Quote Error: {text}")
                return None
            return await resp.json()

    async def get_swap_tx(self, quote_response):
        session = await self._get_session()
        url = f"{self.base_url}/swap"
        payload = {
            "quoteResponse": quote_response,
            "userPublicKey": ExecutionConfig.WALLET_ADDRESS,
            "wrapAndUnwrapSol": True,
            "computeUnitPriceMicroLamports": "auto",
            "prioritizationFeeLamports": "auto"
        }
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                logger.error(f"Jupiter Swap API Error: {text}")
                return None
            data = await resp.json()
            return data.get("swapTransaction")

    async def close(self):
        if self.session:
            await self.session.close()

    @staticmethod
    def deserialize_and_sign(b64_tx_str):
        try:
            tx_bytes = base64.b64decode(b64_tx_str)
            txn = VersionedTransaction.from_bytes(tx_bytes)
            signature = ExecutionConfig.PAYER_KEYPAIR.sign_message(txn.message.to_bytes())
            txn = VersionedTransaction.populate(txn.message, [signature])
            return txn
        except Exception as e:
            logger.error(f"Signing Error: {e}")
            raise