flowchart TD
    A[開始] --> B[画像アップロード]
    B --> C[顔検出処理]
    C --> D{顔を検出?}
    D -->|No| E[エラーメッセージ表示]
    E --> B
    D -->|Yes| F[特徴量抽出]
    F --> G[登録済み顔との照合]
    G --> H{未登録の顔あり?}
    H -->|No| I[結果表示]
    H -->|Yes| J[未登録顔のラベル入力画面]
    J --> K[ラベル登録処理]
    K --> L{登録成功?}
    L -->|No| M[エラーメッセージ表示]
    M --> J
    L -->|Yes| N[登録完了メッセージ]
    N --> I

    subgraph 顔検出処理
    C --> C1[HOGで検出試行]
    C1 --> C2{検出成功?}
    C2 -->|No| C3[CNNで検出試行]
    C2 -->|Yes| C4[検出結果返却]
    C3 --> C4
    end

    subgraph 特徴量抽出処理
    F --> F1[顔画像の正規化]
    F1 --> F2[128次元特徴量抽出]
    F2 --> F3[特徴量の正規化]
    end

    subgraph 照合処理
    G --> G1[登録済み顔データ読み込み]
    G1 --> G2[類似度計算]
    G2 --> G3{閾値以下?}
    G3 -->|Yes| G4[同一人物と判定]
    G3 -->|No| G5[未登録顔と判定]
    end

    subgraph データ保存処理
    K --> K1[一時ファイル作成]
    K1 --> K2[JSONデータ保存]
    K2 --> K3[古い一時ファイル削除]
    end

    %% スタイル定義
    classDef process fill:#f9f,stroke:#333,stroke-width:2px;
    classDef decision fill:#bbf,stroke:#333,stroke-width:2px;
    classDef data fill:#bfb,stroke:#333,stroke-width:2px;
    
    class B,C,F,G,K process;
    class D,H,L decision;
    class I,J,N data;