# 2025-07: Predict the Introverts from the Extroverts — 最終版（vFinal）

Kaggle Playground S5E7 の **二値分類（Introvert / Extrovert）**  
本ノートは、**欠損補完の安定化** → **相互作用＋比率の特徴量設計** → **LGBM / XGB / RF の最適化** → **soft voting（等重み）** で最終モデルを構築。

- **CV（Accuracy）**: **0.9685 ± 0.0023**（StratifiedKFold, n_splits=5, RS=42）  
- **提出ファイル**: `submission_baseline.csv`（`/kaggle/working` に出力）

---

## 1. コンペ情報 / データセット
- Competition: **Playground Series (Season 5, Episode 7)**
- 目的変数: `Personality`（`Extrovert=1`, `Introvert=0`）
- ID列: `id`
- 入力ファイル:  
  - `/kaggle/input/playground-series-s5e7/train.csv`  
  - `/kaggle/input/playground-series-s5e7/test.csv`  
  - `/kaggle/input/playground-series-s5e7/sample_submission.csv`

---

## 2. 方針の全体像（最終版）
1. **EDA（要点だけ）**  
   - 特徴量間の相関が比較的高い  
   - `Stage_fear` と `Drained_after_socializing` の構成が類似
2. **欠損補完（2段構え）**  
   - Cat：`Stage_fear`, `Drained_after_socializing` を **元のカテゴリ分布に従って乱数補完**  
   - Num（歪度が低い）：`Friends_circle_size`, `Post_frequency` は **分布ベース**  
   - Num（他）：`Time_spent_Alone`, `Social_event_attendance`, `Going_outside` を  
     入力 `['Drained_after_socializing','Stage_fear','Friends_circle_size','Post_frequency']` で  
     **RandomForestRegressor による予測補完**
3. **型の明示化**  
   - `Drained_after_socializing`, `Stage_fear` → `int`  
   - `Personality` → **`Int64`（nullable int）**（test側にNaNがあるため）
4. **特徴量設計**  
   - 相互作用（積）6本  
   - **比率（商）4本** 追加（例：`Alone_/_Friends` など）  
   - 比率生成に伴う `inf/-inf` は **NaN に置換 → 各列中央値で補完**（`Personality`を除外）
5. **学習**  
   - LGBM / XGB / RF を **RandomizedSearchCV** で軽く探索（`return_train_score=True`）  
   - それぞれ **最適パラメータで再学習**  
   - **VotingClassifier（soft, 等重み）** でアンサンブル
6. **評価**  
   - StratifiedKFold(5, shuffle=True, RS=42) / `accuracy`  
   - 最終CV：**0.9685 ± 0.0023**

---

## 3. 前処理 / 欠損補完（詳説）

### 3.1 カテゴリの補完（分布ベース）
- `Drained_after_socializing`, `Stage_fear`  
  → 非欠損の **出現割合分布** に従って乱数サンプリングで補完  
  → ラベル構成のバランスを破壊しにくく、単純なモード補完よりも偏りが少ない

### 3.2 数値の補完（分布ベース + 予測）
- **分布ベース**（歪度が低い＝対称性が高い指標として採用）  
  - `Friends_circle_size`, `Post_frequency`
- **予測補完**（非線形関係の活用 / 安定性重視）  
  - 対象：`Time_spent_Alone`, `Social_event_attendance`, `Going_outside`  
  - 入力：`['Drained_after_socializing','Stage_fear','Friends_circle_size','Post_frequency']`  
  - モデル：`RandomForestRegressor(random_state=42)`

### 3.3 比率生成に伴う例外値処理
- 比率系（商）は **分母が 0** になり得る → `inf/-inf` を **NaN に置換**  
- 分布が歪みやすいので **中央値で補完**  
- 補完後は `info()` / `isnull().sum()` で **全列欠損 0** を検証（`Personality` を除く）

---

## 4. 特徴量
### 4.1 相互作用（積）
- `Alone_x_Drained` = `Time_spent_Alone` × `Drained_after_socializing`  
- `Alone_x_StageFear` = `Time_spent_Alone` × `Stage_fear`  
- `Social_x_Drained` = `Social_event_attendance` × `Drained_after_socializing`  
- `Social_x_StageFear` = `Social_event_attendance` × `Stage_fear`  
- `Outside_x_Drained` = `Going_outside` × `Drained_after_socializing`  
- `Outside_x_StageFear` = `Going_outside` × `Stage_fear`

### 4.2 比率（商）★新規
- `Alone_/_Friends` = `Time_spent_Alone` / `Friends_circle_size`  
- `Post_/_Friends` = `Post_frequency` / `Friends_circle_size`  
- `Social_/_Outside` = `Social_event_attendance` / `Going_outside`  
- `Alone_/_Social` = `Time_spent_Alone` / `Social_event_attendance`

> ねらい：絶対量ではなく**相対的な傾向**を表現。  
> ツリーモデルは非線形・相互作用を分割で捉えやすく、上記の情報を取り込みやすい。

---

## 5. 学習設定

- **CV**：`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`  
- **モデルと探索**：  
  - **LightGBM**：`num_leaves`, `max_depth`, `learning_rate` を探索  
  - **XGBoost**：`max_depth`, `learning_rate`, `subsample` を探索  
  - **RandomForest**：`max_depth`, `min_samples_split`, `min_samples_leaf` を探索  
  - すべて `RandomizedSearchCV(..., n_iter=10, scoring='accuracy', return_train_score=True, n_jobs=-1)`
- **最終学習**：各モデルの **best params** で学習し直し  
- **アンサンブル**：`VotingClassifier(voting='soft')`（**等重み**）

> ※ v2 では「train/val ギャップに基づく**重み付き voting**」も試行。  
> 最終版では、**シンプルで頑健**な等重みに回帰。

---

## 6. 結果

- **CV（Accuracy）**：**0.9685 ± 0.0023**  
- 各モデルの `best_params_` / `best_score_` はノート出力を参照  
- 重要度可視化：LGBM / XGB / RF で **feature importances** を出力済み

---

## 7. 再現手順

1. Kaggle Notebook（本リポジトリの `notebook.ipynb`）を開く  
2. 「Save & Run All」  
3. `/kaggle/working/submission_baseline.csv` が生成  
4. 必要に応じて `feature_importances` の図を保存

---

## 8. 変更履歴（v1 → v2 → vFinal）

- **vFinal**（本ノート）
  - **比率系特徴量**を追加（Alone/Friends, Post/Friends, Social/Outside, Alone/Social）
  - 比率生成で発生する `inf/-inf` を **NaN 置換 → 各列中央値で補完**
  - 欠損・型の **工程ごとの検証** を強化（`info()`, `isnull().sum()`）
  - LGBM / XGB / RF を再探索して **最適パラメータ更新**
  - **Voting は soft 等重み**（シンプルで頑健な構成）
  - **CV:** 0.9685 ± 0.0023
- **v2**
  - ydata-profiling による **詳細EDA**（欠損/相関の裏付け強化）
  - カテゴリ→数値の **明示キャスト**（`int` / `Int64`）
  - `RandomizedSearchCV(return_train_score=True)` で **汎化ギャップを可視化**
  - **重み付き soft voting** を試行（train/val ギャップ基準）
- **v1**
  - Cat分布ベース + Num分布ベース/予測補完（RF）の **ベースライン**
  - 相互作用特徴量（積）導入
  - LGBM / XGB / RF の **soft voting（等重み）**

---

