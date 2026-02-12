import os
from dotenv import load_dotenv
from solders.keypair import Keypair


load_dotenv()
class ExecutionConfig:
    '''RPC_URL = os.getenv("QUICKNODE_RPC_URL", "填入RPC地址")
    _PRIV_KEY_STR = os.getenv("SOLANA_PRIVATE_KEY", "")

    if not _PRIV_KEY_STR:
        raise ValueError("Missing SOLANA_PRIVATE_KEY in .env")
    try:
        PAYER_KEYPAIR = Keypair.from_base58_string(_PRIV_KEY_STR)
    except Exception:
        import json
        PAYER_KEYPAIR = Keypair.from_bytes(json.loads(_PRIV_KEY_STR))

    WALLET_ADDRESS = str(PAYER_KEYPAIR.pubkey())'''

    DEFAULT_SLIPPAGE_BPS = 200 # bps
    
    PRIORITY_LEVEL = "High" 
    
    SOL_MINT = "So11111111111111111111111111111111111111112"
    USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"