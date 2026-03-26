# CPU 环境安装指南 (Windows/Linux/Mac)

这个指南适用于没有 NVIDIA GPU 或不想使用 CUDA 的 CPU 环境。

## Windows 安装步骤

### 1. 安装 FFmpeg

**使用 Chocolatey（推荐）：**
```powershell
# 以管理员身份运行 PowerShell
choco install ffmpeg -y
```

**或使用 Scoop：**
```powershell
scoop install ffmpeg
```

**或手动安装：**
- 访问 https://ffmpeg.org/download.html
- 下载 Windows 版本并解压到 `C:\ffmpeg`
- 将 `C:\ffmpeg\bin` 添加到系统 PATH

### 2. 安装 ImageMagick

**使用 Chocolatey：**
```powershell
choco install imagemagick -y
```

**配置 ImageMagick 安全策略：**
编辑 `C:\Program Files\ImageMagick-7.x.x-Q16-HDRI\config\policy.xml`，找到：
```xml
<policy domain="path" rights="none" pattern="@*"/>
```
改为：
```xml
<policy domain="path" rights="read|write" pattern="@*"/>
```

### 3. 创建并激活虚拟环境

```powershell
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境（PowerShell）
.\venv\Scripts\Activate.ps1

# 如果遇到执行策略错误，先运行：
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 4. 安装 CPU 版本的依赖

**重要：先安装 CPU 版本的 PyTorch：**
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**然后安装其他依赖：**
```powershell
pip install -r requirements-cpu.txt
```

### 5. 配置环境变量

创建 `.env` 文件：
```bash
VLLM_API_KEY=local-vllm
VLLM_BASE_URL=http://127.0.0.1:1234/v1
VLLM_MODEL=qwen2.5-72b-instruct
```

## Linux/Mac 安装步骤

### 1. 安装系统依赖

**Ubuntu/Debian:**
```bash
sudo apt install -y ffmpeg libavdevice-dev libavfilter-dev libopus-dev \
  libvpx-dev pkg-config libsrtp2-dev imagemagick
```

**macOS (使用 Homebrew):**
```bash
brew install ffmpeg imagemagick
```

### 2. 配置 ImageMagick（Linux）

```bash
sudo sed -i 's/rights="none" pattern="@\*"/rights="read|write" pattern="@*"/' /etc/ImageMagick-6/policy.xml
```

### 3. 创建并激活虚拟环境

```bash
python3.10 -m venv venv
source venv/bin/activate
```

### 4. 安装 CPU 版本的依赖

```bash
# 先安装 CPU 版本的 PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 然后安装其他依赖
pip install -r requirements-cpu.txt
```

## 验证安装

验证 PyTorch 使用 CPU：
```python
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

应该输出 `CUDA available: False`

## 运行项目

代码会自动检测并使用 CPU 模式运行。Whisper 转录速度会比 GPU 慢，但功能完全正常。

```bash
python main.py
```

## 注意事项

- CPU 模式下，转录速度会明显慢于 GPU 模式（可能慢 5-10 倍）
- 所有功能在 CPU 模式下都可用
- 如果之后想使用 GPU，可以重新安装 `requirements.txt`（包含 CUDA 包）

