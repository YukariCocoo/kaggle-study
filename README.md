# 2025-07: Predict the Introverts from the Extroverts（v1）

Kaggle Playground S5E7 の初版 Notebook。  
分布ベースの欠損補完 + RF予測補完を組み合わせ、  
LightGBM / XGBoost / RandomForest の soft voting でベースラインを構築。

- 特徴量：相互作用（Alone×Drained など）を追加  
- CV：StratifiedKFold(5)  
- 結果：Voting Acc = 0.XXXX ± 0.XXXX（詳細はNotebook）

次の改善予定：欠損補完の安定性検証／歪度対策／CV強化／CatBoostの追加など。
