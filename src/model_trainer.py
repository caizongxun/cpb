import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class LSTMModel(nn.Module):
    """基于PyTorch的LSTM模型架构"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # 全连接层
        self.fc1 = nn.Linear(hidden_size, 32)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # LSTM前向
        lstm_out, (h_n, c_n) = self.lstm(x)
        # 仅使用最后一个LSTM输出
        last_out = lstm_out[:, -1, :]
        
        # 全连接层
        out = self.relu(self.fc1(last_out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class ModelTrainer:
    """模型训练器 - PyTorch + CUDA支持"""
    
    def __init__(self, input_size, hidden_size=128, learning_rate=0.001):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        
        # CUDA模式
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 建立模型
        self.model = LSTMModel(input_size, hidden_size)
        self.model.to(self.device)
        
        # 优化器和损失函数
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # 训练纪录
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.max_patience = 20
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device).float()
            y_batch = y_batch.to(self.device).float().unsqueeze(1)
            
            # 不计算析度
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def evaluate(self, val_loader):
        """验证集证上评估模型"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device).float()
                y_batch = y_batch.to(self.device).float().unsqueeze(1)
                
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss
    
    def fit(self, X_train, y_train, X_val, y_val, epochs=150, batch_size=32):
        """训练模型（含早停機制）"""
        # 创建 DataLoader
        train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        logger.info(f"开始训练: epochs={epochs}, batch_size={batch_size}")
        
        for epoch in range(epochs):
            # 训练
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss = self.evaluate(val_loader)
            self.val_losses.append(val_loss)
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # 早停機制
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                # 保存最优模型
                self._save_checkpoint()
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.max_patience:
                    logger.info(f"早停機制激活（{self.max_patience} epochs没有改进）")
                    break
        
        logger.info("训练完成")
    
    def predict(self, X_test):
        """预测测试集"""
        self.model.eval()
        
        X_test_tensor = torch.from_numpy(X_test).to(self.device).float()
        
        with torch.no_grad():
            predictions = self.model(X_test_tensor)
        
        return predictions.cpu().numpy()
    
    def _save_checkpoint(self):
        """保存模型检查点"""
        checkpoint_dir = Path('./models')
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / 'best_model.pt'
        torch.save({
            'model_state': self.model.state_dict(),
            'input_size': self.input_size,
            'hidden_size': self.hidden_size
        }, checkpoint_path)
    
    def save_model(self, filepath):
        """保存超参数和模型"""
        model_dir = Path(filepath).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        torch.save(self.model.state_dict(), filepath)
        
        # 保存超参数
        config = {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'learning_rate': self.learning_rate,
            'device': str(self.device),
            'best_val_loss': float(self.best_val_loss),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        config_path = model_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"模型已保存: {filepath}")
    
    def load_model(self, filepath):
        """加载模型"""
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        logger.info(f"模型已加载: {filepath}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    trainer = ModelTrainer(input_size=30, hidden_size=128)
    print("模型训练器准备完毕")
    print(f"设备: {trainer.device}")
