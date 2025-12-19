import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import logging
from pathlib import Path
import ta  # Technical Analysis library

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """超过35+技术指标特征工程"""
    
    def __init__(self, lookback_period=200):
        """初始化特征工程师"""
        self.lookback_period = lookback_period
        self.scaler = None
        self.pca = None
        self.feature_names = []
        
    def compute_indicators(self, df):
        """计韶35+技术指标"""
        data = df.copy()
        
        # 1. 价格与成交量 (5)
        data['price_change'] = data['close'].pct_change() * 100
        data['volume_change'] = data['volume'].pct_change() * 100
        data['hl2'] = (data['high'] + data['low']) / 2
        data['hlc3'] = (data['high'] + data['low'] + data['close']) / 3
        data['ohlc4'] = (data['open'] + data['high'] + data['low'] + data['close']) / 4
        
        # 2. 简单移动平均 (5)
        for period in [10, 20, 50, 100, 200]:
            data[f'sma_{period}'] = ta.trend.sma_indicator(data['close'], window=period)
        
        # 3. 指数移动平均 (5)
        for period in [10, 20, 50, 100, 200]:
            data[f'ema_{period}'] = ta.trend.ema_indicator(data['close'], window=period)
        
        # 4. 相对强度指数 (2)
        data['rsi_14'] = ta.momentum.rsi(data['close'], window=14)
        data['rsi_21'] = ta.momentum.rsi(data['close'], window=21)
        
        # 5. MACD (3)
        macd = ta.trend.MACD(data['close'])
        data['macd_line'] = macd.macd()
        data['macd_signal'] = macd.macd_signal()
        data['macd_diff'] = macd.macd_diff()
        
        # 6. 上下事 (2)
        stoch = ta.momentum.StochasticOscillator(data['high'], data['low'], data['close'])
        data['stoch_k'] = stoch.stoch()
        data['stoch_d'] = stoch.stoch_signal()
        
        # 7. 动量 (2)
        data['momentum'] = ta.momentum.roc(data['close'], window=5)
        data['roc_12'] = ta.momentum.roc(data['close'], window=12)
        
        # 8. 布林布非技 (5)
        bb = ta.volatility.BollingerBands(data['close'], window=20, window_dev=2)
        data['bb_upper'] = bb.bollinger_hband()
        data['bb_middle'] = bb.bollinger_mavg()
        data['bb_lower'] = bb.bollinger_lband()
        data['bb_width'] = bb.bollinger_wband()
        data['bb_pband'] = bb.bollinger_pband()
        
        # 9. 真实振幅 (1)
        data['atr_14'] = ta.volatility.average_true_range(data['high'], data['low'], data['close'], window=14)
        
        # 10. ADX指数 (3)
        adx = ta.trend.ADXIndicator(data['high'], data['low'], data['close'], window=14)
        data['adx_14'] = adx.adx()
        data['adx_di_plus'] = adx.adx_pos()
        data['adx_di_minus'] = adx.adx_neg()
        
        # 11. Keltner通道 (3)
        kc = ta.volatility.KeltnerChannel(data['high'], data['low'], data['close'], window=20)
        data['kc_upper'] = kc.keltner_channel_hband()
        data['kc_middle'] = kc.keltner_channel_mband()
        data['kc_lower'] = kc.keltner_channel_lband()
        
        # 12. 规一化ATR (1)
        data['natr'] = ta.volatility.natr(data['high'], data['low'], data['close'], window=14)
        
        # 13. OBV (1)
        data['obv'] = ta.volume.on_balance_volume(data['close'], data['volume'])
        
        # 14. 资金流指数 (2)
        data['cmf'] = ta.volume.chaikin_money_flow(data['high'], data['low'], data['close'], data['volume'], window=20)
        data['mfi'] = ta.volume.money_flow_index(data['high'], data['low'], data['close'], data['volume'], window=14)
        
        # 15. 成交量价格沋 (1)
        data['vpt'] = ta.volume.volume_price_trend(data['close'], data['volume'])
        
        logger.info(f"超过35+技术指标成功")
        return data
    
    def handle_missing_values(self, df):
        """处理缺失值"""
        # 前向填充，然后后向填充
        df = df.fillna(method='ffill')
        df = df.fillna(method='bfill')
        # 仍有NaN则按0填充
        df = df.fillna(0)
        return df
    
    def select_features(self, df, method='correlation', n_features=30):
        """特征选择与降维"""
        # 有效数值列
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # 视述帆上司不需要（作为不会被使用的批量数据）
        if 'timestamp' in numeric_cols:
            numeric_cols.remove('timestamp')
        
        feature_df = df[numeric_cols].copy()
        
        # 算法 1: 相关性分析
        if method == 'correlation':
            corr_matrix = feature_df.corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
            selected_features = [col for col in numeric_cols if col not in to_drop]
            logger.info(f"相关性筛选: 去除{len(to_drop)}个特征, 保留{len(selected_features)}个")
        else:
            selected_features = numeric_cols
        
        # 算法 2: PCA降维
        if len(selected_features) > n_features:
            self.pca = PCA(n_components=n_features)
            feature_df_reduced = self.pca.fit_transform(feature_df[selected_features])
            # 转换为DataFrame
            feature_df = pd.DataFrame(
                feature_df_reduced,
                columns=[f'pca_{i}' for i in range(n_features)],
                index=feature_df.index
            )
            logger.info(f"PCA降维: {len(selected_features)} -> {n_features} 个特征")
            self.feature_names = feature_df.columns.tolist()
        else:
            self.feature_names = selected_features
            feature_df = feature_df[selected_features]
        
        return feature_df
    
    def normalize_features(self, X_train, X_val, X_test, method='minmax'):
        """标准化特征"""
        if method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()
        
        # 仅在训练集上拟合scaler
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"批量正常化: {method}")
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def create_sequences(self, X, y, lookback=60):
        """为时间序列模型创建序列"""
        X_seq, y_seq = [], []
        
        for i in range(len(X) - lookback):
            X_seq.append(X[i:i+lookback])
            y_seq.append(y[i+lookback])
        
        return np.array(X_seq), np.array(y_seq)
    
    def process_dataframe(self, df, target_col='close', lookback=60, test_size=0.15, val_size=0.15):
        """一体化数据处理流程"""
        # 计算技术指标
        df = self.compute_indicators(df)
        
        # 处理缺失值
        df = self.handle_missing_values(df)
        
        # 计算目标变量 (未来7天价格变化百分比)
        df['target'] = df['close'].pct_change(7) * 100
        df = df.dropna()
        
        # 特征选择
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'target']]
        X = df[feature_cols].values
        y = df['target'].values
        
        # 批量正常化
        X_train_idx = int(len(X) * (1 - test_size - val_size))
        X_val_idx = int(len(X) * (1 - test_size))
        
        X_train = X[:X_train_idx]
        X_val = X[X_train_idx:X_val_idx]
        X_test = X[X_val_idx:]
        
        y_train = y[:X_train_idx]
        y_val = y[X_train_idx:X_val_idx]
        y_test = y[X_val_idx:]
        
        X_train_scaled, X_val_scaled, X_test_scaled = self.normalize_features(X_train, X_val, X_test, method='minmax')
        
        # 创建时间序列
        X_train_seq, y_train_seq = self.create_sequences(X_train_scaled, y_train, lookback)
        X_val_seq, y_val_seq = self.create_sequences(X_val_scaled, y_val, lookback)
        X_test_seq, y_test_seq = self.create_sequences(X_test_scaled, y_test, lookback)
        
        logger.info(f"数据执行列成功: Train={len(X_train_seq)}, Val={len(X_val_seq)}, Test={len(X_test_seq)}")
        
        return {
            'X_train': X_train_seq,
            'X_val': X_val_seq,
            'X_test': X_test_seq,
            'y_train': y_train_seq,
            'y_val': y_val_seq,
            'y_test': y_test_seq,
            'feature_names': feature_cols,
            'scaler': self.scaler
        }


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    engineer = FeatureEngineer()
    # 例子: df = pd.read_csv('data/raw_data/BTCUSDT_1h.csv')
    # result = engineer.process_dataframe(df)
    print("特征工程模块准备完毕")
