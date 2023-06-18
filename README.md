# MNIST サンプル

MNIST を手元で実行するサンプル web アプリの学習用です。

## 動作確認済み環境

- Windows11
- WSL2
- Ubuntu20.04
- Poetry1.4.2
- pyenv2.3.17-5-ga57e0b50

## クローン後初回準備

1. .env ファイルに`MONGO_URL`を追加し、MongoDB の接続 URL を追加
2. `pyenv install 3.11.3`で pyenv に python3.11.3 を追加
3. `pyenv local 3.11.3`でこのプロジェクトの pyenv で python3.11.3 を使うように設定
4. `poetry env use 3.11.3`で poetry で python3.11.3 を使うように設定
5. `poetry install`を実行
6. その他 VSCode の設定

## 実行コマンド

1. `poetry shell`で poetry 環境に入る
2. `poetry run python main.py`で学習開始し、.env で設定した MongoDB にモデルが保存される
