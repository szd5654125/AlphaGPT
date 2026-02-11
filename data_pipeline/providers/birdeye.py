import aiohttp
import asyncio
from datetime import datetime, timedelta
from loguru import logger
from data_pipeline.config import Config
from data_pipeline.providers.base import DataProvider

class BirdeyeProvider(DataProvider):
    def __init__(self):
        self.base_url = "https://public-api.birdeye.so"
        self.headers = {
            "X-API-KEY": Config.BIRDEYE_API_KEY,
            "accept": "application/json"
        }
        self.semaphore = asyncio.Semaphore(Config.CONCURRENCY)
        
    async def get_trending_tokens(self, limit=100):
        url = f"{self.base_url}/defi/token_trending"
        params = {
            "sort_by": "rank",
            "sort_type": "asc",
            "offset": "0",
            "limit": str(limit)
        }
        
        async with aiohttp.ClientSession(headers=self.headers) as session:
            try:
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        raw_list = data.get('data', {}).get('tokens', [])
                        
                        results = []
                        for t in raw_list:
                            results.append({
                                'address': t['address'],
                                'symbol': t.get('symbol', 'UNKNOWN'),
                                'name': t.get('name', 'UNKNOWN'),
                                'decimals': t.get('decimals', 6),
                                'liquidity': t.get('liquidity', 0),
                                'fdv': t.get('fdv', 0)
                            })
                        return results
                    else:
                        logger.error(f"Birdeye Trending Error: {resp.status}")
                        return []
            except Exception as e:
                logger.error(f"Birdeye Trending Exception: {e}")
                return []

    async def get_token_history(self, session, address, days=Config.HISTORY_DAYS):
        time_to = int(datetime.now().timestamp())
        time_from = int((datetime.now() - timedelta(days=days)).timestamp())
        
        url = f"{self.base_url}/defi/ohlcv"
        params = {
            "address": address,
            "type": Config.TIMEFRAME,
            "time_from": time_from,
            "time_to": time_to
        }

        async with self.semaphore:
            try:
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        items = data.get('data', {}).get('items', [])
                        if not items: return []
                        
                        formatted = []
                        for item in items:
                            formatted.append((
                                datetime.fromtimestamp(item['unixTime']), # time
                                address,                                  # address
                                float(item['o']),                         # open
                                float(item['h']),                         # high
                                float(item['l']),                         # low
                                float(item['c']),                         # close
                                float(item['v']),                         # volume
                                0.0,                                      # liquidity
                                0.0,                                      # fdv
                                'birdeye'                                 # source
                            ))
                        return formatted
                    elif resp.status == 429:
                        logger.warning(f"Birdeye 429 for {address}, retrying...")
                        await asyncio.sleep(2)
                        return await self.get_token_history(session, address, days)
                    else:
                        return []
            except Exception as e:
                logger.error(f"Birdeye Fetch Error {address}: {e}")
                return []