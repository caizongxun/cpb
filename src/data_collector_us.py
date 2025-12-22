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
        
    def get_timeframe_duration_ms(self, timeframe):
        """計算時間框架的毫秒數"""
        if timeframe == '1m':
            return 60 * 1000
        elif timeframe == '5m':
            return 5 * 60 * 1000
        elif timeframe == '15m':
            return 15 * 60 * 1000
        elif timeframe == '1h':
            return 60 * 60 * 1000
        elif timeframe == '4h':
            return 4 * 60 * 60 * 1000
        elif timeframe == '1d':
            return 24 * 60 * 60 * 1000
        else:
            return 60 * 60 * 1000  # 預設一小時
    
    def fetch_klines_loop(self, symbol, timeframe, min_klines=1000, max_retries=3):
        """迴圈採集 K 線數據，確保量出足最小數量"""
        all_klines = []
        timeframe_duration_ms = self.get_timeframe_duration_ms(timeframe)
        
        logger.info(f"\n正在執行迴圈採集 {symbol} {timeframe} (\u76ee標: {min_klines}+ K\u68d2)...")
        
        # 第一次：從最新的時間開始
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # 計算起始時間
                if len(all_klines) == 0:
                    # 第一邊取最新的數據
                    since = None  # 使用 None 表示從最新檔開始
                    logger.info(f"  採集第 1 批 (1000 K\u68d2)...")
                else:
                    # 後續採集：從最旧的時間開始
                    oldest_time = all_klines[0][0]  # 最旧 K 線的時間戳
                    # 從更早的時間開始，時間每次減少 1000 根 K 線的時間
                    since = oldest_time - (1000 * timeframe_duration_ms)
                    logger.info(f"  採集第 {len(all_klines) // 1000 + 1} 批 ({len(all_klines)} -> {len(all_klines) + 1000} K\u68d2)...")
                
                # 採集數據
                klines = self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe,
                    since=since,
                    limit=1000  # CCXT 每次最大5 1000 根
                )
                
                if not klines:
                    logger.warning(f"  API 返回空數據")
                    break
                
                # 合併 K 線
                if len(all_klines) > 0:
                    # 移除重複數據
                    oldest_existing_time = all_klines[0][0]
                    klines = [k for k in klines if k[0] < oldest_existing_time]
                
                all_klines = klines + all_klines
                
                logger.info(f"    第 {len(klines)} 根 K\u68d2, \u7e3d計 {len(all_klines)} 根")
                
                # 普棄不需需的數據，仅保留 min_klines + 1000
                if len(all_klines) > min_klines + 1000:
                    all_klines = all_klines[:min_klines + 1000]
                    logger.info(f"    扰効\u5230 {len(all_klines)} \u6839")
                
                # 檢查是否已經取得足够
                if len(all_klines) >= min_klines:
                    logger.info(f"  \u2713 \u63a1集完成: {len(all_klines)} K\u68d2 (\u6e1b少 {min_klines} \u6839)")
                    break
                
                # 佴克時間以便下一次請求
                time.sleep(0.5)
                retry_count = 0
                
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    wait_time = 2 ** retry_count
                    logger.warning(f"  API 請求失敗，{wait_time} 秒後重試: {str(e)[:80]}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"  重試 {max_retries} 次後失敗")
                    break
        
        return all_klines
    
    def process_klines(self, klines, symbol, interval):
        """處理 K 線數據為 DataFrame"""
        if not klines:
            logger.warning(f"  沒有獲取到 {symbol} {interval} 的數據")
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
            logger.warning(f"  {symbol} {interval} 存在空值，執行前向填充")
            df = df.fillna(method='ffill')
        
        # 檢查最小數據量
        if len(df) < 1000:
            logger.warning(f"  {symbol} {interval} 數據不足 ({len(df)} < 1000)")
            return None
        
        logger.info(f"  \u2713 成功處理 {symbol} {interval}: {len(df)} K \u68d2")
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
        logger.info(f"  數據已保存: {filepath.name} ({len(df)} 行)")
        return filepath
    
    def collect_all_coins(self, min_klines=1000):
        """採集所有幣種的 K 線數據（確保足量）"""
        end_time = datetime.utcnow()
        
        metadata = {
            'collection_date': datetime.utcnow().isoformat(),
            'collection_time': end_time.isoformat(),
            'source': 'Binance US (CCXT) - Loop-based collection',
            'min_klines_target': min_klines,
            'coins': {},
            'timeframes': self.timeframes
        }
        
        successful_coins = 0
        failed_coins = 0
        
        for coin in self.coins:
            logger.info(f"\n{'='*60}")
            logger.info(f"開始採集: {coin}")
            logger.info(f"{'='*60}")
            
            coin_metadata = {'timeframes': {}}
            coin_success = False
            
            for timeframe in self.timeframes:
                try:
                    # 迴圈採集 K 線（確保足量）
                    klines = self.fetch_klines_loop(
                        coin, 
                        timeframe,
                        min_klines=min_klines,
                        max_retries=3
                    )
                    
                    if not klines:
                        logger.warning(f"  跳過 {coin} {timeframe}: 無數據")
                        continue
                    
                    # 處理數據
                    df = self.process_klines(klines, coin, timeframe)
                    if df is None:
                        logger.warning(f"  跳過 {coin} {timeframe}: 處理失敗")
                        continue
                    
                    # 保存數據
                    filepath = self.save_data(df, coin, timeframe)
                    
                    coin_metadata['timeframes'][timeframe] = {
                        'file': str(filepath),
                        'rows': len(df),
                        'start_date': df['timestamp'].min().isoformat(),
                        'end_date': df['timestamp'].max().isoformat(),
                        'price_range': f"${df['close'].min():.2f} - ${df['close'].max():.2f}"
                    }
                    
                    coin_success = True
                    
                except Exception as e:
                    logger.error(f"  採集 {coin} {timeframe} 失敗: {str(e)[:100]}")
                    continue
            
            if coin_metadata['timeframes']:
                metadata['coins'][coin] = coin_metadata
                successful_coins += 1
            else:
                failed_coins += 1
        
        # 保存元數據
        self._save_metadata(metadata)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"採集完成")
        logger.info(f"{'='*60}")
        logger.info(f"✓ 成功: {successful_coins} 個幣種")
        logger.info(f"✗ 失敗: {failed_coins} 個幣種")
        logger.info(f"成功率: {successful_coins / (successful_coins + failed_coins) * 100:.1f}%")
        
        return metadata
    
    def _save_metadata(self, metadata):
        """保存元數據"""
        metadata_path = self.base_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"\n元數據已保存: {metadata_path}")


if __name__ == '__main__':
    logger.info("使用 Binance US API (迴圈採集模式)")
    collector = BinanceUSDataCollector()
    
    # 迴圈採集，預設每個幣種至少 1000 根 K 線
    metadata = collector.collect_all_coins(min_klines=1000)
    
    print(f"\n✓ 採集完成！")
    print(f"总說: {len(metadata['coins'])} 個幣種")
