# 顔認識プロトタイプ

## 概要
このアプリケーションは、ArcFaceによる顔認識のプロトタイプ機能を含んだリポジトリです。  
`app.py` を実行することで、`static/uploads/test.jpg` にある画像の人物をデータベースと照合し、新規登録または認識結果を表示します。

## セットアップ
1. リポジトリをクローン
2. 仮想環境を作成し、`requirements.txt` をインストール
   ```bash
   python -m venv venv
   source venv/bin/activate  # Mac/Linux
   # Windowsの場合: venv\Scripts\activate
   pip install -r requirements.txt
