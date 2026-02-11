import asyncio
import aiohttp
from loguru import logger
from data_pipeline.config import Config
from data_pipeline.db_manager import DBManager
from data_pipeline.providers.birdeye import BirdeyeProvider
from data_pipeline.providers.dexscreener import DexScreenerProvider

class DataManager:
    def __init__(self):
        self.db = DBManager()
        self.birdeye = BirdeyeProvider()
        self.dexscreener = DexScreenerProvider()
        
    async def initialize(self):
        await self.db.connect()
        await self.db.init_schema()

    async def close(self):
        await self.db.close()

    async def pipeline_sync_daily(self):
        logger.info("Step 1: Discovering trending tokens...")
        limit = 500 if Config.BIRDEYE_IS_PAID else 100
        candidates = await self.birdeye.get_trending_tokens(limit=limit)
        
        logger.info(f"Raw candidates found: {len(candidates)}")

        selected_tokens = []
        for t in candidates:
            liq = t.get('liquidity', 0)
            fdv = t.get('fdv', 0)
            
            if liq < Config.MIN_LIQUIDITY_USD: continue
            if fdv < Config.MIN_FDV: continue
            if fdv > Config.MAX_FDV: continue # 剔除像 WIF/BONK 这种巨无霸，专注于早期高成长
            
            selected_tokens.append(t)
            
        logger.info(f"Tokens selected after filtering: {len(selected_tokens)}")
        
        if not selected_tokens:
            logger.warning("No tokens passed the filter. Relax constraints in Config.")
            return

        db_tokens = [(t['address'], t['symbol'], t['name'], t['decimals'], Config.CHAIN) for t in selected_tokens]
        await self.db.upsert_tokens(db_tokens)

        logger.info(f"Step 4: Fetching OHLCV for {len(selected_tokens)} tokens...")
        
        async with aiohttp.ClientSession(headers=self.birdeye.headers) as session:
            tasks = []
            for t in selected_tokens:
                tasks.append(self.birdeye.get_token_history(session, t['address']))
            
            batch_size = 20
            total_candles = 0
            
            for i in range(0, len(tasks), batch_size):
                batch = tasks[i:i+batch_size]
                results = await asyncio.gather(*batch)
                
                records = [item for sublist in results if sublist for item in sublist]
                
                # 批量写入
                await self.db.batch_insert_ohlcv(records)
                total_candles += len(records)
                logger.info(f"Processed batch {i}/{len(tasks)}. Inserted {len(records)} candles.")
                
        logger.success(f"Pipeline complete. Total candles stored: {total_candles}")