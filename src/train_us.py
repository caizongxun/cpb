import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import torch
from dotenv import load_dotenv

# 導入自訂模組
sys.path.insert(0, str(Path(__file__).parent))
from data_collector_us import BinanceUSDataCollector
from feature_engineer import FeatureEngineer
from model_trainer import ModelTrainer

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 載入環境變數
load_dotenv()

class CryptoMLPipelineUS:
    """完整的加密貨幣ML pipeline - Binance US 版本（無地區限制）"""
    
    def __init__(self, data_dir='./data', model_dir='./models', is_colab=False):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.is_colab = is_colab
        
        # 建立目錄
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"初始化pipeline - Binance US 版本")
        logger.info(f"設備Colab: {is_colab}")
        logger.info(f"訓練設備: PyTorch LSTM + CUDA")
        logger.info(f"GPU 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"GPU 型號: {torch.cuda.get_device_name(0)}")
    
    def phase1_data_collection(self, days_back=30, skip_collection=False):
        """Phase 1: 資料採集 - 使用 Binance US API"""
        logger.info("\n" + "="*60)
        logger.info("Phase 1: 資料採集 (Binance US - 無地區限制)")
        logger.info("="*60)
        
        # 檢查是否已有本地數據
        raw_data_dir = self.data_dir / 'raw_data'
        if skip_collection and raw_data_dir.exists():
            csv_files = list(raw_data_dir.glob('*.csv'))
            if csv_files:
                logger.info(f"跳過採集，使用本地 {len(csv_files)} 個CSV檔案")
                return csv_files
        
        # 採集數據 - 使用 Binance US
        try:
            logger.info("正在使用 Binance US API 採集數據...")
            logger.info("✓ 此 API 無地區限制，無需認證密鑰")
            
            collector = BinanceUSDataCollector()
            metadata = collector.collect_all_coins(days_back=days_back)
            
            logger.info(f"✓ 成功採集 {len(metadata['coins'])} 個幣種")
            return list(raw_data_dir.glob('*.csv'))
        except Exception as e:
            logger.error(f"資料採集失敗: {e}")
            raise
    
    def phase2_feature_engineering(self, csv_files):
        """Phase 2: 特徵工程"""
        logger.info("\n" + "="*60)
        logger.info("Phase 2: 特徵工程 (35+ 技術指標)")
        logger.info("="*60)
        
        processed_data = {}
        engineer = FeatureEngineer(lookback_period=200)
        
        for csv_file in csv_files:
            try:
                logger.info(f"\n處理: {csv_file.name}")
                
                # 讀取CSV
                df = pd.read_csv(csv_file)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                if len(df) < 1000:
                    logger.warning(f"跳過 {csv_file.name}: 資料不足 ({len(df)} < 1000)")
                    continue
                
                # 特徵工程
                result = engineer.process_dataframe(df, lookback=60, test_size=0.15, val_size=0.15)
                processed_data[csv_file.stem] = result
                
                logger.info(f"✓ {csv_file.name} - Train:{result['X_train'].shape[0]}, Val:{result['X_val'].shape[0]}, Test:{result['X_test'].shape[0]}")
                
            except Exception as e:
                logger.error(f"特徵工程失敗 {csv_file.name}: {e}")
                continue
        
        logger.info(f"\n✓ 成功處理 {len(processed_data)} 個幣種")
        return processed_data
    
    def phase3_model_training(self, processed_data, epochs=150, batch_size=32):
        """Phase 3: 模型訓練 (PyTorch + CUDA)"""
        logger.info("\n" + "="*60)
        logger.info("Phase 3: 模型訓練 (PyTorch + CUDA GPU加速)")
        logger.info("="*60)
        
        results = {}
        
        for coin_name, data in processed_data.items():
            logger.info(f"\n訓練 {coin_name}...")
            
            try:
                X_train = data['X_train']
                X_val = data['X_val']
                X_test = data['X_test']
                y_train = data['y_train']
                y_val = data['y_val']
                y_test = data['y_test']
                
                # 初始化訓練器
                trainer = ModelTrainer(
                    input_size=X_train.shape[2],
                    hidden_size=128,
                    learning_rate=0.001
                )
                
                # 訓練
                trainer.fit(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size)
                
                # 測試
                y_pred = trainer.predict(X_test)
                
                # 計算指標
                mse = np.mean((y_pred - y_test.reshape(-1, 1)) ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(y_pred - y_test.reshape(-1, 1)))
                mape = np.mean(np.abs((y_pred - y_test.reshape(-1, 1)) / (y_test.reshape(-1, 1) + 1e-8))) * 100
                
                # 方向準確率
                y_pred_binary = (y_pred > 0).astype(int)
                y_test_binary = (y_test > 0).astype(int)
                direction_accuracy = np.mean(y_pred_binary.flatten() == y_test_binary) * 100
                
                # 儲存模型
                model_path = self.model_dir / f"{coin_name}_lstm.pt"
                trainer.save_model(str(model_path))
                
                results[coin_name] = {
                    'rmse': float(rmse),
                    'mae': float(mae),
                    'mape': float(mape),
                    'direction_accuracy': float(direction_accuracy),
                    'best_val_loss': float(trainer.best_val_loss),
                    'model_path': str(model_path)
                }
                
                logger.info(f"✓ {coin_name}")
                logger.info(f"  RMSE: {rmse:.6f}")
                logger.info(f"  MAE: {mae:.6f}")
                logger.info(f"  MAPE: {mape:.2f}%")
                logger.info(f"  方向準確率: {direction_accuracy:.2f}%")
                
            except Exception as e:
                logger.error(f"訓練失敗 {coin_name}: {e}")
                continue
        
        return results
    
    def phase4_evaluation(self, results):
        """Phase 4: 評估與報告"""
        logger.info("\n" + "="*60)
        logger.info("Phase 4: 模型評估")
        logger.info("="*60)
        
        # 轉為DataFrame查看
        results_df = pd.DataFrame(results).T
        
        logger.info("\n=== 模型性能統計 ===")
        logger.info(f"\n總訓練幣種: {len(results)}")
        logger.info(f"\nRMSE統計:")
        logger.info(f"  平均: {results_df['rmse'].mean():.6f}")
        logger.info(f"  最小: {results_df['rmse'].min():.6f}")
        logger.info(f"  最大: {results_df['rmse'].max():.6f}")
        logger.info(f"\nMAPE統計:")
        logger.info(f"  平均: {results_df['mape'].mean():.2f}%")
        logger.info(f"  最小: {results_df['mape'].min():.2f}%")
        logger.info(f"  最大: {results_df['mape'].max():.2f}%")
        logger.info(f"\n方向準確率統計:")
        logger.info(f"  平均: {results_df['direction_accuracy'].mean():.2f}%")
        logger.info(f"  最小: {results_df['direction_accuracy'].min():.2f}%")
        logger.info(f"  最大: {results_df['direction_accuracy'].max():.2f}%")
        
        # 儲存報告
        report_path = self.model_dir / 'training_report.json'
        with open(report_path, 'w') as f:
            json.dump({
                'timestamp': datetime.utcnow().isoformat(),
                'source': 'Binance US (CCXT)',
                'results': results,
                'summary': {
                    'total_coins': len(results),
                    'avg_rmse': float(results_df['rmse'].mean()),
                    'avg_mape': float(results_df['mape'].mean()),
                    'avg_direction_accuracy': float(results_df['direction_accuracy'].mean())
                }
            }, f, indent=2)
        
        logger.info(f"\n✓ 報告已儲存: {report_path}")
        return results_df
    
    def run_full_pipeline(self, days_back=30, epochs=150, skip_collection=False):
        """執行完整pipeline"""
        logger.info("\n" + "#"*60)
        logger.info("# 開始執行完整ML Pipeline - Binance US 版本")
        logger.info("#"*60)
        logger.info(f"# 日週: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        logger.info(f"# GPU: {torch.cuda.is_available()} - 使用 {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        logger.info("#"*60)
        
        try:
            # Phase 1
            csv_files = self.phase1_data_collection(days_back=days_back, skip_collection=skip_collection)
            
            # Phase 2
            processed_data = self.phase2_feature_engineering(csv_files)
            
            # Phase 3
            results = self.phase3_model_training(processed_data, epochs=epochs)
            
            # Phase 4
            results_df = self.phase4_evaluation(results)
            
            logger.info("\n" + "#"*60)
            logger.info("# Pipeline執行完成！")
            logger.info("#"*60)
            
            return results_df
            
        except Exception as e:
            logger.error(f"Pipeline執行失敗: {e}")
            raise


if __name__ == '__main__':
    # Colab檢測
    try:
        from google.colab import drive
        IS_COLAB = True
        logger.info("偵測到Google Colab環境")
    except ImportError:
        IS_COLAB = False
        logger.info("本地環境")
    
    # Colab環境設置
    if IS_COLAB:
        from google.colab import drive
        drive.mount('/content/drive')
        # 改變工作目錄到Google Drive
        work_dir = '/content/drive/MyDrive/cpb_training'
        os.makedirs(work_dir, exist_ok=True)
        os.chdir(work_dir)
    
    # 初始化Pipeline
    pipeline = CryptoMLPipelineUS(is_colab=IS_COLAB)
    
    # 執行完整pipeline
    results_df = pipeline.run_full_pipeline(
        days_back=30,
        epochs=150,
        skip_collection=False  # 改為True可跳過採集，使用本地數據
    )
    
    # 顯示最終結果
    logger.info("\n=== 最終結果 ===")
    logger.info(results_df)
