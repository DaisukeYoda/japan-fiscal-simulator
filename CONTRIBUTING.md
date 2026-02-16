# コントリビューションガイド

jpfs へのコントリビューションを歓迎します。

## 開発環境のセットアップ

```bash
git clone https://github.com/DaisukeYoda/japan-fiscal-simulator.git
cd japan-fiscal-simulator
uv sync
```

## 開発コマンド

```bash
# テスト
uv run pytest

# 型チェック（strictモード）
uv run mypy src/japan_fiscal_simulator

# Lint & フォーマット
uv run ruff check src tests
uv run ruff format src tests
```

## プルリクエストの手順

1. `main` ブランチから feature ブランチを作成
2. 変更を実装し、テストを追加・通過させる
3. `ruff check`、`ruff format`、`mypy` をパスさせる
4. PRを作成（タイトル・説明は日本語）

## コーディング規約

- すべての関数・メソッドに型ヒントを付ける
- データ構造には dataclass を使用
- 継承よりも Protocol ベースのインターフェースを優先

## Issue の報告

バグ報告や機能リクエストは [Issues](https://github.com/DaisukeYoda/japan-fiscal-simulator/issues) から投稿してください。
