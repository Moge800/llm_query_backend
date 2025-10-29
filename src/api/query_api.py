"""
レガシー互換性のための エントリーポイント
新しいクリーンアーキテクチャ版を使用してください: src.main
"""

# 新しいクリーンアーキテクチャ版のアプリケーションをインポート
from ..main import app

# レガシー互換性のために、元の関数やクラスをここで再エクスポートすることも可能
# ただし、新しいアーキテクチャへの移行を推奨します

if __name__ == "__main__":
    import uvicorn
    import os

    uvicorn.run(
        app, host=os.getenv("HOST", "localhost"), port=int(os.getenv("PORT", 8010))
    )
