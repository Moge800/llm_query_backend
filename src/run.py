"""アプリケーションエントリーポイント"""

import os
from .main import app

if __name__ == "__main__":
    import uvicorn

    # 環境変数から設定を取得（main.pyのConfigと重複しますが、実行時の便宜上）
    host = os.getenv("HOST", "localhost")
    port = int(os.getenv("PORT", 8010))

    uvicorn.run(app, host=host, port=port)
