# Binance 地區限制錯誤解決方案

## 錯誤訊息

```
APIError(code=0): Service unavailable from a restricted location according to 'b. Eligibility'
```

**這不是 API Token 的問題！** 這是 Binance 根據你的 **IP 地址** 判斷你所在的地區是否被限制。

---

## 原因分析

### 為什麼會出現這個錯誤？

1. **Binance 限制的地區**
   - 某些國家/地區被 Binance 禁止使用
   - Binance 根據 IP 地址判斷地理位置
   - 如果檢測到你在限制地區，API 會拒絕請求

2. **Colab IP 地址問題**
   - Google Colab 的伺服器 IP 可能被標記為受限地區
   - 或者 IP 地址不穩定，被多個地區檢測

3. **VPN/代理影響**
   - 使用了某些 VPN 或代理，被 Binance 認為是受限地區

---

## 解決方案

### 方案 1: 使用公開數據（推薦）

**不需要 API 密鑰！** 直接使用 Binance 公開 API：

```python
# 修改 src/data_collector.py

# 注釋掉這部分
# self.client = Client(self.api_key, self.api_secret)

# 改用公開 API（不需要認證）
import requests

class BinanceDataCollector:
    def __init__(self, api_key=None, api_secret=None):
        # 不使用認證的 Client
        self.session = requests.Session()
        self.base_url = 'https://api.binance.com/api'
        
    def fetch_klines_public(self, symbol, interval, limit=1000, startTime=None, endTime=None):
        """使用公開 API（無需認證）"""
        url = f'{self.base_url}/v3/klines'
        
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        if startTime:
            params['startTime'] = startTime
        if endTime:
            params['endTime'] = endTime
        
        response = self.session.get(url, params=params)
        
        if response.status_code != 200:
            logger.error(f"API 請求失敗: {response.text}")
            return []
        
        return response.json()
```

**優點**：
- ✓ 不受地區限制
- ✓ 無需 API 密鑰
- ✓ 完全免費
- ✓ 速度快

**缺點**：
- ✗ 請求限制較多（每分鐘 1200 個請求）
- ✗ 某些進階功能不可用

### 方案 2: 使用 VPN（不推薦）

在 Colab 中配置 VPN：

```bash
# 安裝 OpenVPN
!apt-get update && apt-get install -y openvpn

# 連接到支持的地區的 VPN
!openvpn --config /path/to/vpn/config.ovpn --daemon

# 等待連接
import time
time.sleep(5)
```

**風險**：
- ✗ Binance 可能偵測到 VPN 並拒絕
- ✗ VPN 連接不穩定
- ✗ 速度變慢

### 方案 3: 在本機訓練（最佳）

如果你在支持的地區，在本機電腦上運行：

```bash
# 本機
python src/train.py
```

**優點**：
- ✓ 無地區限制
- ✓ IP 是你的本機 IP
- ✓ 速度快
- ✓ 穩定可靠

**缺點**：
- ✗ 需要本機有 GPU（可選）
- ✗ 訓練速度較慢（無 GPU）

### 方案 4: 使用替代數據源

使用其他不受限制的數據源：

#### 選項 A: CoinGecko API
```python
import requests

def fetch_from_coingecko(coin_id, vs_currency='usd', days=30):
    url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart'
    params = {
        'vs_currency': vs_currency,
        'days': days,
        'interval': 'daily'
    }
    response = requests.get(url, params=params)
    return response.json()

# 使用示例
data = fetch_from_coingecko('bitcoin', days=30)
print(data)
```

#### 選項 B: CCXT 庫（支持多個交易所）
```python
!pip install ccxt

import ccxt

# 使用不受限制的交易所
exchange = ccxt.kraken()  # 或 binanceus, coinbase 等
data = exchange.fetch_ohlcv('BTC/USD', '1h', limit=100)
```

#### 選項 C: Alpha Vantage
```python
import requests

def fetch_crypto_data(api_key):
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'CURRENCY_EXCHANGE_RATE',
        'from_currency': 'BTC',
        'to_currency': 'USD',
        'apikey': api_key
    }
    response = requests.get(url, params=params)
    return response.json()
```

---

## 快速修復：使用公開 API 版本

我已經為你準備了 **公開 API 版本**，完全不受地區限制。

### 改用公開 API 的完整代碼

在 Colab 中執行這個 Cell：

```python
# 1. 先卸載舊的 python-binance
!pip uninstall -y python-binance

# 2. 安裝 ccxt（支持多個交易所，無地區限制）
!pip install -q ccxt

# 3. 替換數據採集函數
import ccxt
import pandas as pd
from datetime import datetime, timedelta

def fetch_klines_public(symbol, timeframe='1h', limit=3000, days=30):
    """
    使用 CCXT 公開 API 採集 K 線數據
    不需要 API 密鑰，無地區限制
    """
    try:
        # 使用 Binance 公開 API（通過 CCXT）
        exchange = ccxt.binance({'enableRateLimit': True})
        
        # 將符號格式從 BTCUSDT 改為 BTC/USDT
        exchange_symbol = symbol.replace('USDT', '/USDT')
        
        # 計算起始時間
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)
        start_ms = int(start_time.timestamp() * 1000)
        
        print(f'正在採集 {symbol} {timeframe}...')
        
        all_candles = []
        since = start_ms
        
        while since < int(end_time.timestamp() * 1000):
            try:
                candles = exchange.fetch_ohlcv(
                    exchange_symbol,
                    timeframe,
                    since=since,
                    limit=limit
                )
                
                if not candles:
                    break
                
                all_candles.extend(candles)
                
                # 更新起始時間到最後一根 K 線
                since = candles[-1][0] + 1
                
                print(f'  已採集 {len(all_candles)} 根 K 線...')
                
            except Exception as e:
                print(f'  採集出錯: {e}')
                break
        
        return all_candles
        
    except Exception as e:
        print(f'錯誤: {e}')
        return []

# 測試
data = fetch_klines_public('BTCUSDT', '1h', limit=500, days=7)
print(f'成功採集 {len(data)} 根 K 線')

# 轉換為 DataFrame
df = pd.DataFrame(
    data,
    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
)
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

print(df.head())
print(f'時間範圍: {df["timestamp"].min()} 到 {df["timestamp"].max()}')
```

---

## 檢查你的位置

在 Colab 中檢查你的 IP 地址和位置：

```python
import requests

# 檢查 IP 地址
response = requests.get('https://api.ipify.org?format=json')
ip = response.json()['ip']
print(f'你的 IP: {ip}')

# 檢查地理位置
response = requests.get(f'https://ipapi.co/{ip}/json')
geo = response.json()
print(f'國家: {geo.get("country_name")}')
print(f'城市: {geo.get("city")}')
print(f'ISP: {geo.get("org")}')
```

---

## Binance 禁止國家清單

Binance 目前限制以下地區（可能更新）：

- 美國（需使用 Binance US）
- 加拿大（某些省份）
- 香港
- 新加坡
- 日本
- 澳大利亞
- 歐盟（需 KYC 驗證）
- 某些中東和非洲國家

**完整清單**: https://www.binance.com/en/support/faq/360038955691

---

## 推薦解決方案排序

| 優先級 | 方案 | 難度 | 速度 | 說明 |
|------|------|------|------|------|
| 🥇 **1** | 使用公開 API（CCXT） | 簡單 | 快 | 推薦，無限制，無需密鑰 |
| 🥈 **2** | 本機訓練 | 中等 | 中 | 如果本機在支持地區 |
| 🥉 **3** | CoinGecko / Alpha Vantage | 簡單 | 快 | 數據可用性有限 |
| 4️⃣ | VPN | 複雜 | 慢 | 風險高，可能被偵測 |

---

## 檢查清單

在重新運行訓練前，檢查這些：

- [ ] 確認你在支持的國家（或使用 VPN/代理）
- [ ] API 密鑰是否有效（如果使用認證 API）
- [ ] 建議使用公開 API（無需密鑰，無地區限制）
- [ ] 檢查 Colab IP 是否被限制
- [ ] 考慮在本機或支持地區的伺服器上運行

---

## 完整的無限制版本

我會為你準備一個 **完全無限制的版本**，使用公開 API。

你現在要做的是：

1. **複製上面的公開 API 代碼**
2. **在 Colab 中執行，測試是否成功採集數據**
3. **如果成功，我會更新整個 Pipeline 使用公開 API**

---

## 立即修復（3 步）

### Step 1: 在 Colab 中執行

```python
!pip install -q ccxt
import ccxt

exchange = ccxt.binance()
data = exchange.fetch_ohlcv('BTC/USDT', '1h', limit=100)
print(f'成功採集 {len(data)} 根 K 線！')
print(f'最新價格: ${data[-1][4]}')
```

### Step 2: 檢查是否成功

如果輸出顯示採集成功，說明 **公開 API 可用**。

### Step 3: 使用公開 API 版本訓練

```python
# 改用公開 API 的 Pipeline
pipeline = CryptoMLPipeline(use_public_api=True)
results = pipeline.run_full_pipeline()
```

---

## 常見問題

### Q: API 密鑰是否過期？
**A**: 不會。API 密鑰永遠有效，除非你手動刪除。問題是 **地區限制**，不是密鑰。

### Q: 為什麼本機可以但 Colab 不行？
**A**: 因為 Colab 使用 Google 的伺服器 IP，該 IP 可能被 Binance 標記為受限地區。

### Q: 公開 API 會不會被限制？
**A**: 公開 API 無需認證，Binance 難以追蹤，被限制的機率很小。

### Q: 我應該選擇哪個方案？
**A**: 
- 如果你在支持的國家 → 用本機或 VPN
- 如果你在限制國家 → 用公開 API 或 CCXT
- 如果你不確定 → 先試公開 API（最安全）

---

好消息：**你的 API 密鑰和程式碼都沒問題！** 只是需要換一種方法採集數據。

建議立即試試 CCXT 公開 API，應該會成功！
