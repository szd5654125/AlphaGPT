from loguru import logger
from data_pipeline.providers.base import DataProvider
from config.general_config import DatabaseConfig

class DexScreenerProvider(DataProvider):
    def __init__(self):
        self.base_url = "https://api.dexscreener.com/latest/dex"
    
    async def get_trending_tokens(self, limit=50):
        url = f"https://api.dexscreener.com/latest/dex/tokens/solana" 
        return []

    async def get_token_details_batch(self, session, addresses):
        valid_data = []
        chunk_size = 30
        
        for i in range(0, len(addresses), chunk_size):
            chunk = addresses[i:i+chunk_size]
            addr_str = ",".join(chunk)
            url = f"{self.base_url}/tokens/{addr_str}"
            
            try:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        pairs = data.get('pairs', [])
                        
                        best_pairs = {}
                        for p in pairs:
                            if p['chainId'] != DatabaseConfig.CHAIN: continue
                            base_addr = p['baseToken']['address']
                            liq = float(p.get('liquidity', {}).get('usd', 0))
                            
                            if base_addr not in best_pairs or liq > best_pairs[base_addr]['liquidity']:
                                best_pairs[base_addr] = {
                                    'address': base_addr,
                                    'symbol': p['baseToken']['symbol'],
                                    'name': p['baseToken']['name'],
                                    'liquidity': liq,
                                    'fdv': float(p.get('fdv', 0)),
                                    'decimals': 6 # 默认
                                }
                        valid_data.extend(best_pairs.values())
            except Exception as e:
                logger.error(f"DexScreener batch error: {e}")
        
        return valid_data

    async def get_token_history(self, session, address, days):
        return []