# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

Dolphinは文書画像解析用のマルチモーダルVLMモデル（0.3B）です。2段階のanalyze-then-parseアプローチで、ページレベルのレイアウト解析と要素レベルの並列パースを行います。

## 開発環境のセットアップ（uv使用）

このプロジェクトはPythonで書かれており、uvを使用して依存関係を管理します。

### ステップ1: 仮想環境の作成

**重要:** Python 3.11を使用してください（torch 2.1.0はPython 3.12に対応していません）

```bash
# Python 3.11で仮想環境を作成
uv venv --python 3.11
```

### ステップ2: 依存関係のインストール

requirements.txtには互換性の問題があるため、以下のコマンドで個別にインストールします：

```bash
# 互換性のあるバージョンでインストール
uv pip install numpy==1.24.4 omegaconf==2.3.0 opencv-python==4.5.5.64 opencv-python-headless==4.5.5.64 pillow==9.3.0 timm==0.5.4 torch==2.1.0 torchvision==0.16.0 transformers==4.47.0 accelerate==1.6.0 pymupdf==1.26
```

**注意点:**
- `opencv-python==4.11.0.86`（requirements.txt記載）は`numpy==1.24.4`と互換性がないため、`4.5.5.64`を使用
- `torch==2.1.0`はPython 3.11以下が必要

### ステップ3: GPU版PyTorchのインストール（オプションだが推奨）

NVIDIA GPU（RTX 4070 Ti SUPERなど）を使用する場合、CUDA版をインストール：

```bash
# CPU版をアンインストール
uv pip uninstall torch torchvision

# CUDA 12.1版をインストール
uv pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
```

**GPU動作確認:**
```bash
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

### ステップ4: モデルのダウンロード

**推奨: Hugging Face形式**
```bash
# Hugging Face CLIでダウンロード（huggingface_hubは既にインストール済み）
uv run hf download ByteDance/Dolphin --local-dir ./hf_model
```

これで約263MBのモデルファイル（model.safetensors他）が`./hf_model`ディレクトリにダウンロードされます。

**代替: Git LFSでダウンロード**
```bash
git lfs install
git clone https://huggingface.co/ByteDance/Dolphin ./hf_model
```

**オプション: オリジナル形式（config-based）**
```bash
# checkpoints/ディレクトリに以下を配置:
# - dolphin_model.bin
# - dolphin_tokenizer.json
# Baidu Yun または Google Drive からダウンロード
```

### インストールの依存関係
- Python: 3.11（必須）
- PyTorch: 2.1.0 + torchvision 0.16.0
- transformers: 4.47.0
- timm: 0.5.4
- opencv-python: 4.5.5.64
- numpy: 1.24.4
- omegaconf: 2.3.0
- pillow: 9.3.0
- pymupdf: 1.26
- accelerate: 1.6.0

## よく使うコマンド

### ページレベル解析（単一画像）

**オリジナルフレームワーク:**
```bash
uv run python demo_page.py --config ./config/Dolphin.yaml --input_path ./demo/page_imgs/page_1.jpeg --save_dir ./results
```

**Hugging Face版:**
```bash
uv run python demo_page_hf.py --model_path ./hf_model --input_path ./demo/page_imgs/page_1.jpeg --save_dir ./results
```

### PDF処理
```bash
uv run python demo_page_hf.py --model_path ./hf_model --input_path ./demo/page_imgs/page_6.pdf --save_dir ./results
```

### ディレクトリ内の全ファイルを処理
```bash
uv run python demo_page_hf.py --model_path ./hf_model --input_path ./demo/page_imgs --save_dir ./results --max_batch_size 16
```

### 要素レベル解析

**テーブル解析:**
```bash
uv run python demo_element_hf.py --model_path ./hf_model --input_path ./demo/element_imgs/table_1.jpeg --element_type table
```

**数式解析:**
```bash
uv run python demo_element_hf.py --model_path ./hf_model --input_path ./demo/element_imgs/line_formula.jpeg --element_type formula
```

**テキスト解析:**
```bash
uv run python demo_element_hf.py --model_path ./hf_model --input_path ./demo/element_imgs/para_1.jpg --element_type text
```

### 実行結果の確認

処理結果は`save_dir`で指定したディレクトリに保存されます（デフォルト: 入力と同じディレクトリ）

**ファイル構造:**
```
results/
├── recognition_json/
│   └── [ファイル名].json      # 構造化データ（bbox, label, text, reading_order）
└── markdown/
    ├── [ファイル名].md         # Markdown形式の結果
    └── figures/                # 抽出された図
        └── [ファイル名]_figure_*.png
```

**例: PDFを処理した場合**
```bash
uv run python demo_page_hf.py --model_path ./hf_model --input_path ./demo/page_imgs/page_6.pdf --save_dir ./results
```

結果:
- `results/recognition_json/page_6.json` - 全9ページの解析結果（JSON）
- `results/markdown/page_6.md` - 全9ページの内容（Markdown）
- `results/markdown/figures/page_6_page_00X_figure_*.png` - 抽出された図

## アーキテクチャ概要

### コア構造

1. **2段階パイプライン** (`demo_page.py` / `demo_page_hf.py`):
   - Stage 1: `model.chat("Parse the reading order of this document.", image)` でレイアウト解析
   - Stage 2: `process_elements()` で個別要素を並列パース

2. **モデル実装**:
   - `chat.py`: オリジナル版のDOLPHINクラス（config-based）
   - `demo_page_hf.py`: Hugging Face版のDOLPHINクラス（AutoProcessor使用）
   - `utils/model.py`: SwinEncoder + BARTDecoderのコアモデル
   - `utils/processor.py`: 画像前処理（896x896リサイズ、パディング、正規化）

3. **推論フロー**:
   ```
   入力画像 → 896x896にリサイズ+パディング → SwinEncoder
     → プロンプト（"Parse reading order..."）→ BARTDecoder
     → レイアウト結果パース → 各要素を切り出し
     → バッチで並列推論（テキスト/テーブル別）→ 結果統合
   ```

### デバイスとデータ型
- CUDA利用可能時: float16（GPU）
- CPU使用時: float32
- `demo_page_hf.py:34-37` で自動判定

### バッチ処理
- `max_batch_size`: 要素レベルの並列デコード時のバッチサイズ（デフォルト: オリジナル版=4, HF版=16）
- HF版は要素タイプ（text/table）ごとにグループ化してバッチ処理（`demo_page_hf.py:250-263`）

### 出力形式
- JSON: 各要素のbbox、label、text、reading_order
- Markdown: 結合されたテキスト（図は`![Figure](figures/...)`形式）
- 図は `save_dir/figures/` に保存

### 主要設定
- `config/Dolphin.yaml`: モデル設定（オリジナル版）
  - Swinパラメータ: img_size=[896,896], patch_size=4, embed_dim=128
  - Decoder: 10層, hidden_dim=1024, max_length=4096

## デプロイメント

高速化版も利用可能:
- vLLM: `deployment/vllm/`
- TensorRT-LLM: `deployment/tensorrt_llm/`

詳細は各ディレクトリのReadMe.mdを参照。

## コードフォーマット

pyproject.tomlにBlack設定あり（line-length: 120）。

## トラブルシューティング

### 依存関係のエラー

**問題: `torch==2.1.0`がインストールできない**
```
× No solution found when resolving dependencies:
  ╰─▶ Because torch==2.1.0 has no wheels with a matching Python ABI tag
```

**解決策:** Python 3.11を使用してください
```bash
uv venv --python 3.11
```

**問題: `opencv-python`と`numpy`のバージョン競合**
```
× opencv-python==4.11.0.86 depends on numpy>=1.26.0 and you require numpy==1.24.4
```

**解決策:** opencv-python 4.5.5.64を使用
```bash
uv pip install opencv-python==4.5.5.64 opencv-python-headless==4.5.5.64
```

### 実行時のエラー

**問題: `Input type (struct c10::Half) and bias type (float) should be the same`**

CPUで実行時にfloat16/float32の型不一致が発生。

**解決策:** このエラーは修正済みです。最新のコードでは自動的にCPU/GPUに応じて適切な型を使用します。

**問題: GPU（CUDA）が認識されない**

**確認:**
```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```

**解決策:** CUDA版PyTorchをインストール
```bash
uv pip uninstall torch torchvision
uv pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
```

### パフォーマンス

**GPU VRAMが不足する場合**

`max_batch_size`を小さくしてください：
```bash
uv run python demo_page_hf.py --model_path ./hf_model --input_path input.pdf --max_batch_size 4
```

デフォルト値:
- オリジナル版: 4
- HF版: 16

**処理が遅い場合**

1. GPU版PyTorchを使用しているか確認
2. CUDA（float16）で実行されているか確認
3. バッチサイズを増やす（VRAMに余裕がある場合）
