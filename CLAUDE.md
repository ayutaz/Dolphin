# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

Dolphinは文書画像解析用のマルチモーダルVLMモデル（0.3B）です。2段階のanalyze-then-parseアプローチで、ページレベルのレイアウト解析と要素レベルの並列パースを行います。

## 開発環境のセットアップ（uv使用）

このプロジェクトはPythonで書かれており、uvを使用して依存関係を管理します。

```bash
# uvで仮想環境を作成して依存関係をインストール
uv venv
uv pip install -r requirements.txt
```

### 必要な依存関係
- PyTorch 2.1.0 + torchvision 0.16.0
- transformers 4.47.0
- timm 0.5.4
- opencv-python, pillow, pymupdf
- omegaconf 2.3.0

### モデルのダウンロード

**オプションA: オリジナル形式（config-based）**
```bash
# checkpoints/ディレクトリに以下を配置:
# - dolphin_model.bin
# - dolphin_tokenizer.json
# Baidu Yun または Google Drive からダウンロード
```

**オプションB: Hugging Face形式**
```bash
# Hugging Face Hubからモデルをダウンロード
git lfs install
git clone https://huggingface.co/ByteDance/Dolphin ./hf_model
# または
uv pip install huggingface_hub
huggingface-cli download ByteDance/Dolphin --local-dir ./hf_model
```

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
