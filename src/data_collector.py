import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from binance.client import Client
from dotenv import load_dotenv
import logging
from pathlib import Path
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class BinanceDataCollector:
    """采集币安加密货币K线数据"""
    
    def __init__(self, api_key=None, api_secret=None):
        """初始化币安客户端"""
        self.api_key = api_key or os.getenv('BINANCE_API_KEY')
        self.api_secret = api_secret or os.getenv('BINANCE_API_SECRET')
        
        if not self.api_key or not self.api_secret:
            logger.warning("API密钥未设置，仅使用公开数据")
        
        self.client = Client(self.api_key, self.api_secret)
        self.base_dir = Path(os.getenv('DATA_DIR', './data'))
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # 20+ 主流币种
        self.coins = [
            'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'ADAUSDT',
            'DOGEUSDT', 'XRPUSDT', 'POLKADOT', 'DOTUSDT', 'LINKUSDT',
            'UNIUSDT', 'AVAXUSDT', 'MATICUSDT', 'LTCUSDT', 'ETCUSDT',
            'BCHABCUSDT', 'XLMUSDT', 'FILUSDT', 'AXSUSDT', 'MANAUSDT',
            'GRTUSDT', 'CRVUSDT'
        ]
        
        self.timeframes = ['15m', '1h']
        
    def fetch_klines(self, symbol, interval, limit=3000, start_time=None, end_time=None):
        """获取K线数据，支持时间范围"""
        all_klines = []
        current_start = start_time
        max_retries = 3
        retry_count = 0
        
        try:
            while retry_count < max_retries:
                try:
                    logger.info(f"正在获取 {symbol} {interval} 数据...")
                    klines = self.client.get_historical_klines(
                        symbol,
                        interval,
                        start_str=current_start,
                        end_str=end_time,
                        limit=limit
                    )
                    
                    if not klines:
                        break
                    
                    all_klines.extend(klines)
                    
                    # 如果获取数量少于limit，说明已到达数据末尾
                    if len(klines) < limit:
                        break
                    
                    # 更新start_time为最后一根K线的时间
                    current_start = klines[-1][0] + 1
                    time.sleep(0.1)  # 避免API速率限制
                    retry_count = 0
                    
                except Exception as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        wait_time = 2 ** retry_count  # 指数退避
                        logger.warning(f"API请求失败，{wait_time}秒后重试: {str(e)}")
                        time.sleep(wait_time)
                    else:
                        raise
            
            return all_klines
        
        except Exception as e:
            logger.error(f"获取 {symbol} {interval} 数据失败: {str(e)}")
            return []
    
    def process_klines(self, klines, symbol, interval):
        """处理K线数据为DataFrame"""
        if not klines:
            logger.warning(f"没有获取到 {symbol} {interval} 的数据")
            return None
        
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # 数据类型转换
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 去重和排序
        df = df.drop_duplicates(subset=['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 数据验证
        if df.isnull().any().any():
            logger.warning(f"{symbol} {interval} 存在空值，执行前向填充")
            df = df.fillna(method='ffill')
        
        # 检查最小数据量
        if len(df) < 1000:
            logger.warning(f"{symbol} {interval} 数据不足 ({len(df)} < 1000)")
            return None
        
        logger.info(f"成功处理 {symbol} {interval}: {len(df)} K棒")
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    def save_data(self, df, symbol, interval):
        """保存数据为CSV"""
        output_dir = self.base_dir / 'raw_data'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{symbol}_{interval}.csv"
        filepath = output_dir / filename
        
        df.to_csv(filepath, index=False)
        logger.info(f"数据已保存: {filepath}")
        return filepath
    
    def collect_all_coins(self, days_back=30):
        """采集所有币种的指定时间段数据"""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days_back)
        
        start_str = int(start_time.timestamp() * 1000)
        end_str = int(end_time.timestamp() * 1000)
        
        metadata = {
            'collection_date': datetime.utcnow().isoformat(),
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'coins': {},
            'timeframes': self.timeframes
        }
        
        for coin in self.coins:
            logger.info(f"\n========== 开始采集 {coin} ==========")
            coin_metadata = {'timeframes': {}}
            
            for timeframe in self.timeframes:
                try:
                    # 获取K线
                    klines = self.fetch_klines(
                        coin, 
                        timeframe,
                        start_time=start_str,
                        end_time=end_str
                    )
                    
                    if not klines:
                        logger.warning(f"跳过 {coin} {timeframe}: 无数据")
                        continue
                    
                    # 处理数据
                    df = self.process_klines(klines, coin, timeframe)
                    if df is None:
                        continue
                    
                    # 保存数据
                    filepath = self.save_data(df, coin, timeframe)
                    
                    coin_metadata['timeframes'][timeframe] = {
                        'file': str(filepath),
                        'rows': len(df),
                        'start_date': df['timestamp'].min().isoformat(),
                        'end_date': df['timestamp'].max().isoformat()
                    }
                    
                except Exception as e:
                    logger.error(f"采集 {coin} {timeframe} 失败: {str(e)}")
                    continue
            
            if coin_metadata['timeframes']:
                metadata['coins'][coin] = coin_metadata
        
        # 保存元数据
        self._save_metadata(metadata)
        logger.info(f"\n采集完成！共成功采集 {len(metadata['coins'])} 个币种")
        
        return metadata
    
    def _save_metadata(self, metadata):
        """保存元数据"""
        metadata_path = self.base_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"元数据已保存: {metadata_path}")


if __name__ == '__main__':
    collector = BinanceDataCollector()
    metadata = collector.collect_all_coins(days_back=30)
    print("\n采集完成！")
    print(f"总币种数: {len(metadata['coins'])}")
    for coin, info in metadata['coins'].items():
        print(f"  {coin}: {list(info['timeframes'].keys())}")
