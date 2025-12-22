# Google Colab 中的 .env 檔案管理指南

## .env 檔案在 Colab 中的位置

### 工作目錄結構

```
/content/drive/MyDrive/cpb_training/
│
├── cpb/                    ← Clone 下來的專案
│   ├── .env                ← .env 檔案在這裡！
│   ├── src/
│   ├── models/
│   ├── data/
│   └── ...
│
└── 其他檔案...
```

### 絕對路徑

```
/content/drive/MyDrive/cpb_training/cpb/.env
```

---

## 方法 1: 直接在 Colab 中編輯 .env (推薦)

### Step 1: 打開文件編輯器

在 Colab Cell 中執行：

```python
# 在 Colab 中打開 .env 檔案
with open('.env', 'r') as f:
    content = f.read()
    print(content)
```

### Step 2: 修改 .env 檔案

執行這個 Cell 來編輯：

```python
# 讀取現有 .env
with open('.env', 'r') as f:
    lines = f.readlines()

# 修改 API 密鑰
for i, line in enumerate(lines):
    if line.startswith('BINANCE_API_KEY='):
        lines[i] = 'BINANCE_API_KEY=your_actual_api_key_here\n'
    elif line.startswith('BINANCE_API_SECRET='):
        lines[i] = 'BINANCE_API_SECRET=your_actual_secret_here\n'

# 寫入修改
with open('.env', 'w') as f:
    f.writelines(lines)

print('✓ .env 已更新！')

# 驗證修改
with open('.env', 'r') as f:
    print(f.read())
```

**將這個替換為你的實際值：**
- `your_actual_api_key_here` → 你的 Binance API Key
- `your_actual_secret_here` → 你的 Binance API Secret

### Step 3: 驗證配置已加載

執行：

```python
from dotenv import load_dotenv
import os

# 重新加載 .env
load_dotenv()

# 驗證
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')

if api_key and api_secret:
    print(f"✓ API Key 已配置: {api_key[:10]}...***")
    print(f"✓ API Secret 已配置: {api_secret[:10]}...***")
else:
    print("✗ API 密鑰未配置或未加載")
```

---

## 方法 2: 使用 Google Drive 直接編輯

### Step 1: 打開 Google Drive

1. 進入 [Google Drive](https://drive.google.com)
2. 進入資料夾：`MyDrive > cpb_training > cpb`
3. 找到 `.env` 檔案

### Step 2: 用 Google Drive 文字編輯器編輯

1. 右鍵點擊 `.env` 檔案
2. 選擇 `打開方式 > Google Drive 文字編輯器`
3. 修改內容：
   ```env
   # 修改這兩行
   BINANCE_API_KEY=your_actual_key_here
   BINANCE_API_SECRET=your_actual_secret_here
   ```
4. 儲存 (Ctrl+S 或 Cmd+S)

### Step 3: 重新載入 Colab 中的 .env

在 Colab 執行：

```python
from dotenv import load_dotenv
import os

# 強制重新加載 .env
load_dotenv('.env', override=True)

print('✓ .env 已重新加載')
```

---

## 方法 3: 在 Colab 中直接貼貼密鑰 (臨時)

如果你想避免修改檔案，可以在 Colab 中直接設定環境變量：

```python
import os

# 直接設定環境變量（僅限此 Colab session）
os.environ['BINANCE_API_KEY'] = 'your_actual_key_here'
os.environ['BINANCE_API_SECRET'] = 'your_actual_secret_here'

print('✓ 環境變量已設定')
```

**注意**：重新啟動 Colab 後會遺失，僅臨時有效。

---

## 完整的 Colab 設定步驟

### Step-by-Step 指南

#### 1. 在 Colab 中執行初始化 Cell

```python
from google.colab import drive
import os
import subprocess

# 掛載 Google Drive
drive.mount('/content/drive')

# 進入工作目錄
work_dir = '/content/drive/MyDrive/cpb_training'
os.makedirs(work_dir, exist_ok=True)
os.chdir(work_dir)

print(f'✓ 工作目錄: {os.getcwd()}')
```

#### 2. Clone Repo

```python
if not os.path.exists('cpb'):
    subprocess.run(['git', 'clone', 'https://github.com/caizongxun/cpb.git'], cwd=work_dir)
else:
    subprocess.run(['git', 'pull'], cwd=os.path.join(work_dir, 'cpb'))

os.chdir(os.path.join(work_dir, 'cpb'))
print(f'✓ 當前目錄: {os.getcwd()}')
print(f'✓ 檔案列表: {os.listdir()}')
```

#### 3. 檢視 .env 位置

```python
print(f'✓ .env 完整路徑: {os.path.abspath(".env")}')
print(f'✓ 當前工作目錄: {os.getcwd()}')

# 檢視 .env 內容
if os.path.exists('.env'):
    with open('.env', 'r') as f:
        print('\n=== .env 檔案內容 ===')
        print(f.read())
else:
    print('✗ .env 檔案不存在')
```

#### 4. 修改 .env 中的 API 密鑰

```python
# 方法 A: 自動替換
api_key = 'your_binance_api_key_here'  # ← 填入你的 API Key
api_secret = 'your_binance_api_secret_here'  # ← 填入你的 Secret

with open('.env', 'r') as f:
    content = f.read()

# 替換 API 密鑰
content = content.replace(
    'BINANCE_API_KEY=your_api_key_here',
    f'BINANCE_API_KEY={api_key}'
)
content = content.replace(
    'BINANCE_API_SECRET=your_api_secret_here',
    f'BINANCE_API_SECRET={api_secret}'
)

# 寫入
with open('.env', 'w') as f:
    f.write(content)

print('✓ .env 已更新！')
print('\n=== 更新後的 .env ===')
with open('.env', 'r') as f:
    print(f.read())
```

#### 5. 驗證 API 密鑰已加載

```python
from dotenv import load_dotenv
import os

# 重新加載環境變量
load_dotenv('.env', override=True)

# 驗證
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')

if api_key and api_key != 'your_api_key_here':
    print(f'✓ BINANCE_API_KEY: {api_key[:10]}...{api_key[-4:]}')
else:
    print('✗ BINANCE_API_KEY 未正確設定')

if api_secret and api_secret != 'your_api_secret_here':
    print(f'✓ BINANCE_API_SECRET: {api_secret[:10]}...{api_secret[-4:]}')
else:
    print('✗ BINANCE_API_SECRET 未正確設定')
```

---

## 常見問題

### Q1: 修改 .env 後，訓練還是說找不到 API Key?

**A**: 需要重新加載環境變量：

```python
from dotenv import load_dotenv
import os

# 強制重新加載
load_dotenv('.env', override=True)
```

### Q2: .env 檔案在哪裡?

**A**: 當前工作目錄 (cwd) 中。執行以檢查：

```python
import os
print(f'當前目錄: {os.getcwd()}')
print(f'.env 路徑: {os.path.abspath(".env")}')
print(f'.env 存在: {os.path.exists(".env")}')
```

### Q3: 可以不設定 API Key 嗎?

**A**: 可以。不設定也能運行，但會有 API 請求限制。推薦設定。

### Q4: API Key 會被洩露嗎?

**A**: 
- .env 已加入 `.gitignore`，不會被 push 到 GitHub
- 只存在你的 Google Drive 中
- Colab session 結束後會保留在 Drive，下次 Colab 會重新讀取

### Q5: 修改 .env 後需要重啟 Colab 嗎?

**A**: 不需要。執行上面的重新加載 Cell 即可。

---

## 安全建議

1. **絕對不要** push .env 到 GitHub
   - 已在 `.gitignore` 中排除
   - 但務必確認不會洩露

2. **定期檢查** API Key 的使用情況
   - 進入 Binance 帳戶檢查
   - 如有異常立即重置

3. **使用 IP 白名單**
   - 在 Binance API 管理頁面設定
   - 限制只能從特定 IP 存取

4. **使用讀取權限 API**
   - 建議建立獨立的 API Key
   - 只給予「讀取」權限（不給「交易」）

---

## 快速參考

### .env 檔案路徑
```
Colab 工作目錄下: ./cpb/.env
Google Drive: MyDrive > cpb_training > cpb > .env
絕對路徑: /content/drive/MyDrive/cpb_training/cpb/.env
```

### 修改 API Key 的 3 種方式

| 方式 | 位置 | 難度 | 說明 |
|------|------|------|------|
| **Colab Cell** | 直接在 Python 中修改 | 簡單 | 推薦，即改即用 |
| **Google Drive 編輯** | 在 Drive 中打開檔案 | 中等 | 需要重新加載 |
| **環境變量** | 直接設定 os.environ | 簡單 | 臨時有效，重啟遺失 |

---

## 完整示例代碼

把這個粘貼到 Colab，一次性完成所有設定：

```python
from google.colab import drive
import os
import subprocess
from dotenv import load_dotenv

# ===== 1. 掛載 Drive =====
drive.mount('/content/drive')

# ===== 2. 設定工作目錄 =====
work_dir = '/content/drive/MyDrive/cpb_training'
os.makedirs(work_dir, exist_ok=True)
os.chdir(work_dir)

# ===== 3. Clone Repo =====
if not os.path.exists('cpb'):
    subprocess.run(['git', 'clone', 'https://github.com/caizongxun/cpb.git'])
os.chdir('cpb')

# ===== 4. 設定 API 密鑰 =====
API_KEY = 'your_binance_api_key'          # ← 改這裡
API_SECRET = 'your_binance_api_secret'    # ← 改這裡

# 讀取 .env
with open('.env', 'r') as f:
    lines = f.readlines()

# 修改 API 密鑰行
for i, line in enumerate(lines):
    if 'BINANCE_API_KEY=' in line:
        lines[i] = f'BINANCE_API_KEY={API_KEY}\n'
    elif 'BINANCE_API_SECRET=' in line:
        lines[i] = f'BINANCE_API_SECRET={API_SECRET}\n'

# 寫入
with open('.env', 'w') as f:
    f.writelines(lines)

# ===== 5. 驗證 =====
load_dotenv('.env', override=True)

key = os.getenv('BINANCE_API_KEY')
secret = os.getenv('BINANCE_API_SECRET')

print(f'✓ 工作目錄: {os.getcwd()}')
print(f'✓ .env 路徑: {os.path.abspath(".env")}')
print(f'✓ API Key 已設定: {key[:10]}...' if key != 'your_binance_api_key' else '✗ API Key 未設定')
print(f'✓ API Secret 已設定: {secret[:10]}...' if secret != 'your_binance_api_secret' else '✗ API Secret 未設定')
```

修改上面代碼中的 `your_binance_api_key` 和 `your_binance_api_secret`，然後執行！

---

## 總結

✓ **.env 在 Colab 的位置**: `./cpb/.env`（相對路徑）或 `/content/drive/MyDrive/cpb_training/cpb/.env`（絕對路徑）  
✓ **修改方式**: 直接在 Colab Cell 中用 Python 修改（推薦）  
✓ **驗證方式**: 執行 `load_dotenv()` 並檢查 `os.getenv()`  
✓ **安全**: .env 不會被 push 到 GitHub，只存在於你的 Google Drive  

祝設定順利！
