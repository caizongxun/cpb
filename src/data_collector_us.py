import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
import logging
from pathlib import Path
import time
import ccxt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class BinanceUSDataCollector:
    """採集 Binance US 加密貨幣 K 線數據（無地區限制）"""
    
    def __init__(self):
        """初始化 Binance US 客戶端"""
        try:
            # 使用 CCXT 連接 Binance US（無需 API 密鑰，無地區限制）
            self.exchange = ccxt.binanceus({
                'enableRateLimit': True,
                'rateLimit': 500  # 避免速率限制
            })
            logger.info("✓ 已連接到 Binance US API")
        except Exception as e:
            logger.error(f"無法連接到 Binance US: {e}")
            raise
        
        self.base_dir = Path(os.getenv('DATA_DIR', './data'))
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # 20+ 主流幣種
        self.coins = [
            'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'ADA/USDT',
            'DOGE/USDT', 'XRP/USDT', 'DOT/USDT', 'LINK/USDT', 'UNI/USDT',
            'AVAX/USDT', 'MATIC/USDT', 'LTC/USDT', 'ETC/USDT', 'BCH/USDT',
            'XLM/USDT', 'FIL/USDT', 'AXS/USDT', 'MANA/USDT', 'GRT/USDT',
            'CRV/USDT'
        ]
        
        self.timeframes = ['15m', '1h']
        
    def fetch_klines(self, symbol, timeframe, limit=1000, days_back=30):
        """從 Binance US 獲取 K 線數據"""
        all_klines = []
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days_back)
        
        since = int(start_time.timestamp() * 1000)  # 毫秒時間戳
        max_retries = 3
        retry_count = 0
        
        try:
            while retry_count < max_retries:
                try:
                    logger.info(f"正在獲取 {symbol} {timeframe} 數據...")
                    
                    # 使用 CCXT 的 fetch_ohlcv
                    klines = self.exchange.fetch_ohlcv(
                        symbol,
                        timeframe,
                        since=since,
                        limit=limit
                    )
                    
                    if not klines:
                        break
                    
                    all_klines.extend(klines)
                    logger.info(f"獲取了 {len(klines)} 根 K 線，總計 {len(all_klines)} 根")
                    
                    # 如果獲取數量少於 limit，說明已到達終點
                    if len(klines) < limit:
                        break
                    
                    # 更新起始時間到最後一根 K 線之後
                    since = int(klines[-1][0]) + 1
                    time.sleep(0.1)  # 避免速率限制
                    retry_count = 0
                    
                except Exception as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        wait_time = 2 ** retry_count
                        logger.warning(f"API 請求失敗，{wait_time} 秒後重試: {str(e)}")
                        time.sleep(wait_time)
                    else:
                        raise
            
            return all_klines
        
        except Exception as e:
            logger.error(f"獲取 {symbol} {timeframe} 數據失敗: {str(e)}")
            return []
    
    def process_klines(self, klines, symbol, interval):
        """處理 K 線數據為 DataFrame"""
        if not klines:
            logger.warning(f"沒有獲取到 {symbol} {interval} 的數據")
            return None
        
        # CCXT 返回格式: [timestamp, open, high, low, close, volume]
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume'
        ])
        
        # 轉換時間戳
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # 數據類型轉換
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 去重和排序
        df = df.drop_duplicates(subset=['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 數據驗證
        if df.isnull().any().any():
            logger.warning(f"{symbol} {interval} 存在空值，執行前向填充")
            df = df.fillna(method='ffill')
        
        # 檢查最小數據量
        if len(df) < 1000:
            logger.warning(f"{symbol} {interval} 數據不足 ({len(df)} < 1000)")
            return None
        
        logger.info(f"成功處理 {symbol} {interval}: {len(df)} K 棒")
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    def save_data(self, df, symbol, interval):
        """保存數據為 CSV"""
        output_dir = self.base_dir / 'raw_data'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 移除 / 符號以創建合法的文件名
        safe_symbol = symbol.replace('/', '')
        filename = f"{safe_symbol}_{interval}.csv"
        filepath = output_dir / filename
        
        df.to_csv(filepath, index=False)
        logger.info(f"數據已保存: {filepath}")
        return filepath
    
    def collect_all_coins(self, days_back=30):
        """採集所有幣種的指定時間段數據"""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days_back)
        
        metadata = {
            'collection_date': datetime.utcnow().isoformat(),
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'source': 'Binance US (CCXT)',
            'coins': {},
            'timeframes': self.timeframes
        }
        
        for coin in self.coins:
            logger.info(f"\n========== 開始採集 {coin} ==========")
            coin_metadata = {'timeframes': {}}
            
            for timeframe in self.timeframes:
                try:
                    # 獲取 K 線
                    klines = self.fetch_klines(coin, timeframe, days_back=days_back)
                    
                    if not klines:
                        logger.warning(f"跳過 {coin} {timeframe}: 無數據")
                        continue
                    
                    # 處理數據
                    df = self.process_klines(klines, coin, timeframe)
                    if df is None:
                        continue
                    
                    # 保存數據
                    filepath = self.save_data(df, coin, timeframe)
                    
                    coin_metadata['timeframes'][timeframe] = {
                        'file': str(filepath),
                        'rows': len(df),
                        'start_date': df['timestamp'].min().isoformat(),
                        'end_date': df['timestamp'].max().isoformat()
                    }
                    
                except Exception as e:
                    logger.error(f"採集 {coin} {timeframe} 失敗: {str(e)}")
                    continue
            
            if coin_metadata['timeframes']:
                metadata['coins'][coin] = coin_metadata
        
        # 保存元數據
        self._save_metadata(metadata)
        logger.info(f"\n採集完成！共成功採集 {len(metadata['coins'])} 個幣種")
        
        return metadata
    
    def _save_metadata(self, metadata):
        """保存元數據"""
        metadata_path = self.base_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"元數據已保存: {metadata_path}")


if __name__ == '__main__':
    logger.info("使用 Binance US API（無地區限制）")
    collector = BinanceUSDataCollector()
    metadata = collector.collect_all_coins(days_back=30)
    print("\n採集完成！")
    print(f"總幣種數: {len(metadata['coins'])}")
    for coin, info in list(metadata['coins'].items())[:5]:
        print(f"  {coin}: {list(info['timeframes'].keys())}")
