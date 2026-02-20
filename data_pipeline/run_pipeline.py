import asyncio
from loguru import logger
from data_pipeline.data_manager import DataManager
from config.general_config import DatabaseConfig

async def main():
    if not DatabaseConfig.BIRDEYE_API_KEY:
        logger.error("BIRDEYE_API_KEY is missing in .env")
        return

    manager = DataManager()
    
    try:
        await manager.initialize()
        await manager.pipeline_sync_daily()
    except Exception as e:
        logger.exception(f"Pipeline crashed: {e}")
    finally:
        await manager.close()

if __name__ == "__main__":
    asyncio.run(main())