from solana.rpc.async_api import AsyncClient
from solders.pubkey import Pubkey
from execution.config import ExecutionConfig
import torch

async def get_mint_decimals(mint_str: str, client: AsyncClient) -> int:
    if mint_str == ExecutionConfig.SOL_MINT:
        return 9
    try:
        pubkey = Pubkey.from_string(mint_str)
        resp = await client.get_account_info(pubkey)
        if resp.value is None:
            return 6
        resp_parsed = await client.get_account_info_json_parsed(pubkey)
        decimals = resp_parsed.value.data.parsed['info']['decimals']
        return int(decimals)
    except Exception:
        return 6

def shift1(x, fill=0.0):
    # x: [N, T]
    pad = torch.full((x.shape[0], 1), fill, device=x.device, dtype=x.dtype)
    return torch.cat([pad, x[:, :-1]], dim=1)
