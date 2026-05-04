# IEEE-CIS Fraud Detection

## Kaggle-ის კონკურსის მოკლე მიმოხილვა

[IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) არის თაღლითური ტრანზაქციების ამოცნობის კონკურსი, რომელიც Vesta Corporation-ის რეალურ e-commerce მონაცემებს იყენებს. მონაცემებში წარმოდგენილია მიახლოებით 590,000 სასწავლო ტრანზაქცია და 507,000 ტესტ ტრანზაქცია, 430 feature-ით.

- **სამიზნე ცვლადი**: `isFraud` (binary, ~3.5% positive class — imbalanced).
- **შეფასების მეტრიკა**: **Area Under the ROC Curve (AUC)**.
- **გამოწვევები**: მაღალი mismatch train/test შორის (test set მომავალშია), V*/D* feature-ების მნიშვნელოვანი ნაწილი NaN-ია, კატეგორიული feature-ების დიდი ცვალებადობა (`card1`-ს ~13K უნიკალური მნიშვნელობა აქვს).

## ჩემი მიდგომა პრობლემის გადასაჭრელად

ექსპერიმენტი შესრულდა ორ ფაზად:

1. **ფაზა 1 — საბაზისო შემოწმება**: გავტესტე 6 სხვადასხვა მოდელის არქიტექტურა ერთი და იმავე საბაზო preprocessing-ით (`fillna(-999)`, frequency encoding, hour/day/log_amount), რათა გამომევლინა საუკეთესო კანდიდატი. მოდელებს შორის XGBoost აშკარა ლიდერი იყო (val_auc 0.9561 vs შემდეგი საუკეთესო RandomForest 0.9128).

2. **ფაზა 2 — XGBoost-ის ოპტიმიზაცია**: საუკეთესო მოდელზე ჩავატარე ღრმა მუშაობა — UID-feature engineering, time-based cross-validation, Optuna hyperparameter search 25 ცდით და სრული Pipeline, რომელიც პირდაპირ raw test set-ზე მუშაობს.

## რეპოზიტორიის სტრუქტურა

```
MLassignment2/
├── README.md                                      # ფაილი, რომელიც დეტალურად აღწერს პროექტსა და მასზე მუშაობის სრულ პროცესს. 
├── model-experiment-LogisticRegression.ipynb      # მოდელი ლოჯისტიკური რეგრესიით
├── model-experiment-DescisionTree.ipynb           # მოდელი გადაწყვეტილების ხეებით
├── model-experiment-RandomForest.ipynb            # მოდელი რანდომ ფორესტით
├── model-experiment-AdaBoost.ipynb                # მოდელი ადა-ბუსტით
├── model-experiment-MLP.ipynb                     # მოდელი MLP-ით
├── model-experiment-XGBoost.ipynb                 # ოპტიმიზებული მოდელი XGBoost-ით(საბოლოო არჩევანი)
├── model_inference.ipynb                          # საუკეთესო მოდელის ინფერენსი test set-ზე
```

### ყველა ფაილის განმარტება

- **`model-experiment-LogisticRegression.ipynb`** — წრფივი მოდელი StandardScaler-ით, L2 რეგულარიზაციით (`C=0.1`).
- **`model-experiment-DescisionTree.ipynb`** — ერთი გადაწყვეტილების ხე, შემდეგ `max_depth=10`-ით რეგულარიზებული.
- **`model-experiment-RandomForest.ipynb`** — 100/200/150 ხეები, `max_depth=12/20`, ფიჩერების შერჩევით (top50).
- **`model-experiment-AdaBoost.ipynb`** — 50 stump, `max_depth=1`.
- **`model-experiment-MLP.ipynb`** — ერთფენიანი ნეირონული ქსელი 64 ნეირონით, ReLU აქტივაციით, StandardScaler-ით.
- **`model-experiment-XGBoost.ipynb`** — სრულად ოპტიმიზირებული XGBoost: UID engineering + UID aggregations, time-based holdout, Optuna 25 trials, საბოლოო ფაიფლაინი.
- **`model_inference.ipynb`** — საუკეთესო მოდელის (`XGBoost_FraudDetection`) ჩატვირთვა MLflow Model Registry-დან და `submission.csv`-ის გენერაცია.

---

## Feature Engineering

### კატეგორიული ცვლადების რიცხვითში გადაყვანა

ყველა საბაზისო მოდელში გამოყენებულია **frequency encoding** — თითოეული კატეგორიული მნიშვნელობა შეიცვალა მისი შესაბამისი სიხშირით ტრენინგ მონაცემებში:

ოპტიმიზებული XGBoost-ში დავამატე:

- **Frequency + Count encoding** ერთდროულად — Pipeline-ის `FrequencyCountEncoder` ცვლის ორიგინალ სვეტს frequency-ით და ამატებს `<col>_count` სვეტს იმავე კატეგორიის რაოდენობით. 
- **Email parsing** — `P_emaildomain` და `R_emaildomain` დაიყო `_suffix`-ად (com/net/ru/...) და `_bin`-ად (gmail/yahoo/...).
- **Device parsing** — `DeviceInfo` → `DeviceInfo_family` (samsung/iphone/...); `id_30` → `OS_family`; `id_31` → `browser_family`; `id_33` → `screen_w`/`screen_h`.

### NaN მნიშვნელობების დამუშავება

დატაში ძალიან ბევრი NაN არის `D` და `V` სვეტებში, დაახლოებით 30%-90%-მდე missing rate.

- **საბაზისო მოდელები**: `df.fillna(-999)` — ყველაფერი ჩანაცვლდა −999-ით, რათა ხის მოდელებს ცალკე "missing" branch-იც გაეთვალისწინებინათ. წრფივი მოდელებისთვის (LogReg, MLP) ეს მცდარი მიდგომა იყო, რადგან −999 outlier-ად აღიქმება.
- **ოპტიმიზებული XGBoost**: NaN-ები **შენარჩუნდა** — XGBoost ადგილობრივად იყენებს NaN-ს როგორც learnable split direction-ს, ეს უფრო ინფორმატიულია ვიდრე −999-ით ჩანაცვლება.

### Cleaning მიდგომები

XGBoost-ში დავამატე **`reduce_mem_usage`** — ყველა numeric სვეტი დაიყვანება უმცირეს dtype-ზე (`int64 → int8/16/32`, `float64 → float32`). ეფექტი RAM-ის გამოყენების შემცირება, რაც საშუალებას იძლეოდა Kaggle-ის kernel-ზე ყველა(430) feature-ით ერთდროულად მუშაობა memory error-ის გარეშე. ეს გადამწყვეტი იყო, რადგან UID aggregations-ის შემდეგ feature-ების რაოდენობა 600-მდე გაიზარდა.

---

## Feature Selection

### გამოყენებული მიდგომები და მათი შეფასება

**1. Constant / near-constant სვეტების ამოღება** — `nunique <= 1` სვეტები ამოღებულია ყველა მოდელისთვის. ეფექტი: - 5-10 უსარგებლო სვეტი.

**2. Frequency encoding ყველა object სვეტისთვის** — შემდეგ `nunique > 1` ფილტრი. საბაზისო მოდელებში ეს იყო ერთადერთი feature selection-ის ეტაპი.

**3. RandomForest top-50 features** — RandomForest feature importance-ით საუკეთესო 50 feature ავიღე. შედეგი: 0.9128 → 0.9126 (ფაქტობრივად განსხვავება არ ჩანს). დასკვნა: top-N filtering მცირე რაოდენობის feature-ებში მხოლოდ უსარგებლო noise-ს აშორებს, მაგრამ tree ensemble ისედაც კარგად უმკლავდება ამას.

**4. NaN-ratio threshold pruning**  — `nan_ratio > 0.95` სვეტების ამოკლება. ეფექტი: ~30 სვეტი (V-ების ყველაზე "ცარიელი" ნაწილი). მცირე გაუმჯობესება პლუს სტაბილურობა.

**5. XGBoost zero-importance pruning**  — სწრაფი XGBoost (300 ხე) ვატრენინგე და `feature_importances_ == 0` სვეტები ამოვიღე. შედეგი: zero-importance pruning პრაქტიკულად არ ცვლის val_auc-ს, რადგან XGBoost ისედაც აიგნორებს ასეთ სვეტებს. საბოლოო ფაიფლაინში გამოვიყენე `USE_IMPORTANCE_PRUNED=False`, ანუ მთლიანი matrix მივეცი მოდელს.

**6. UID-based aggregations** — feature engineering-ისა და selection-ის ჰიბრიდი. `card1`, `card1+addr1`, `card1+addr1+P_emaildomain`, `card1+addr1+D1n` (სადაც `D1n = day - D1`) — ოთხი UID. თითოეული UID-ისთვის გამოვთვალე mean/std `TransactionAmt`, `D15`, `id_02`, `C13`, `C14`, `D4`-ისთვის. ეს გვაძლევს 48 ახალ feature-ს, რომლებიც "ერთი მომხმარებლის ქცევას" ასახავენ. **ეს იყო ყველაზე ეფექტური ნაბიჯი.**

---

## Training

### ტესტირებული მოდელები

ყველა საბაზისო მოდელი დატრენინგდა stratified random 80/20 train/val split-ზე, `random_state=42`-ით:

| მოდელი | Train AUC | Val AUC | Train−Val gap | ანალიზი |
|---|---|---|---|---|
| **LogisticRegression** (`C=0.1`) | 0.8528 | 0.8502 | 0.003 | **Underfit** — train და val თანაბრად სუსტი. წრფივი მოდელი სუსტია ამ პრობლემისთვის: ფიჩერებს რთული nonlinear ურთიერთქმედებები აქვთ, რომელსაც ლოჯისტიკური რეგრესია ვერ აღიქვამს. |
| **DecisionTree** (`max_depth=∞`) | 1.0000 | 0.7933 | **0.207** | **მძიმე Overfit** — train ზე სრული მორგებაა, val-ზე კი 0.79. ერთი ხე უბრალოდ იზეპირებს ტრენინგ მონაცემებს. |
| **DecisionTree** (`max_depth=10`) | 0.8204 | 0.8134 | 0.007 | **რეგულარიზაციის შემდეგ** overfit შესუსტდა, მაგრამ val_auc 0.81-მდე ჩამოვიდა — ერთი ხე სუსტია. |
| **RandomForest** (`max_depth=∞`, n=100) | 1.0000 | 0.9318 | **0.068** | **Overfit, მაგრამ ბევრი ხის ხარჯზე მაინც კარგად მუშაობს**. |
| **RandomForest** (`max_depth=20`, n=150) | 0.9462 | 0.9110 | 0.035 | **მცირე overfit, კარგი ბალანსი**. მე-2 საუკეთესო მოდელი. |
| **RandomForest top50** (იგივე+top50 features) | 0.9529 | 0.9126 | 0.040 | feature pruning-ის გავლენა მცირეა. |
| **AdaBoost** (`stumps`, n=50, lr=1.0) | 0.8578 | 0.8564 | 0.001 | **Underfit** — depth=1 stump-ები ვერ იჭერენ ფიჩერებს შორის ურთიერთქმედებებს. |
| **MLP** (`hidden=(64,)`, scaler) | 0.8745 | 0.8621 | 0.012 | **Underfit** — MLP ცხრილოვან მონაცემებში ხეების გაერთიანებებს ჩამოუვარდება, განსაკუთრებით როცა NaN-ები ბევრია. |
| **MLP tuned** (`hidden=(128,64)`) | 0.8970 | 0.8819 | 0.015 | მცირე გაუმჯობესება, ისევ underfit. |
| **XGBoost v1** (n=200, depth=6, lr=0.1) | 0.9529 | 0.9386 | 0.014 | კარგად დაბალანსებული. |
| **XGBoost v2 tuned** (n=400, depth=8, lr=0.05) | 0.9780 | 0.9553 | 0.023 | მცირე overfit, მაგრამ val გაიზარდა. |
| **XGBoost Pipeline (ფაზა 1)** | 0.9771 | **0.9561** | 0.021 | სქაუკეთესო საბაზისო მოდელი |

**Overfit/Underfit ანალიზი:**

- **მკვეთრი underfit**: LogisticRegression, AdaBoost (depth=1), MLP. ყველას train_auc < 0.90 — მოდელის capacity არ ჰყოფნის.
- **მკვეთრი overfit**: DecisionTree (depth=∞), RandomForest (depth=∞). train ≈ 1.0, val საგრძნობლად ნაკლები.
- **კარგი ბალანსი**: RandomForest depth=20, ყველა XGBoost.

**ოპტიმიზებული XGBoost** — time-based holdout-ით (test-ის უფრო რეალური ანალოგი) val_auc = **0.9348**, train_auc = **0.9976**. Train-val gap აქ უფრო დიდია (~0.06), მაგრამ ეს არის *honest* gap — რეალური generalization error სავარაუდოდ საბაზისო მოდელებშიც უფრო დიდი იყო, ეს holdout-ით უბრალოდ უკეთ ჩანს.

### Hyperparameter ოპტიმიზაციის მიდგომა

**საბაზისო** manual 1-2 კონფიგურაცია თითოეული მოდელისთვის (`v1`, `v2_tuned`).

**ოპტიმიზებული XGBoost** გამოვიყენე **Optuna 25 trials TPE sampler-ით**, time-based holdout val_auc-ის მაქსიმიზაციით. ძიების სივრცე:

| Hyperparameter | Range |
|---|---|
| `max_depth` | 5–12 |
| `learning_rate` | 0.01–0.1 (log-scale) |
| `subsample` | 0.6–1.0 |
| `colsample_bytree` | 0.3–1.0 |
| `min_child_weight` | 1–100 |
| `gamma` | 0.0–5.0 |
| `reg_alpha` | 1e-8–10 (log) |
| `reg_lambda` | 1e-8–10 (log) |

თითოეული trial 2000 ხეს იყენებდა და `early_stopping_rounds=50`-ით ატრენინგებდა val_auc-ის მონიტორინგით. ყველა trial დაფიქსირდა MLflow-ზე როგორც `XGBoost_Training_Optuna`-ს nested run-ი (`trial_0`...`trial_24`).

დამატებით გავატარე **3-fold time-based cross-validation** (expanding window) საბოლოო კონფიგურაციის სტაბილურობის გადასამოწმებლად — ფოლდების შორის std მცირე იყო.

### საბოლოო მოდელის შერჩევის დასაბუთება

**XGBoost ოპტიმიზებული Pipeline** აშკარად აღემატება ყველა სხვა კანდიდატს:

| მოდელი | Val AUC | Public LB | Private LB |
|---|---|---|---|
| საბაზისო XGBoost (`XGBoost_Pipeline_v1`) | 0.9561 (random split) | **0.933920**| **0.902645**  |
| **ოპტიმიზებული XGBoost (`XGBoost_Pipeline_Final`)** | **0.9348 (time-based)** | **0.9466** | **0.9163** |

ოპტიმიზებული მოდელის Kaggle Leaderboard-ის შედეგი საბაზისოს უსწრებს, რაც ადასტურებს, რომ:

1. **UID engineering + aggregations** აძლიერებს ფიჩერების სიგნალს — fraud rings-ი ერთი და იმავე UID-ის ქვეშ მრავალჯერ ჩნდება.
2. **Time-based holdout** წარმოაჩენს რეალურ generalization-ს (val_auc 0.9348 ≈ private LB 0.9163, ხოლო random split-ის 0.9561 საბაზისოში leakage-ის გამო ოპტიმისტური იყო).
3. **Optuna search** პოულობს ისეთ hyperparameter-ებს, რომელთა ნახვა მანუალურად ვერ მოხერხდებოდა.
4. **NaN-ის შენარჩუნება** XGBoost missing-handling-ით უკეთესია, ვიდრე −999 fillna მიდგომით.

ფინალური მოდელი დარეგისტრირდა MLflow Model Registry-ში სახელით **`XGBoost_FraudDetection`** — `model_inference.ipynb` სწორედ აქედან ტვირთავს მას და უშვებს raw test set-ზე ერთი `predict_proba` გამოძახებით.

---

## MLflow Tracking

### MLflow ექსპერიმენტების ბმული
Dagshub Repository: https://dagshub.com/smama23/MLassignment2
MLflow: https://dagshub.com/smama23/MLassignment2.mlflow/

ექსპერიმენტები:

- `LogisticRegression_Training`
- `DecisionTree_Training`
- `RandomForest_Training`
- `AdaBoost_Training`
- `MLP_Training`
- `XGBoost_Training` — შეიცავს როგორც საბაზისო (`XGBoost_Cleaning_v1`, `XGBoost_FeatureEngineering_v1`, `XGBoost_FeatureSelection_v1`, `XGBoost_Training_v1/v2_tuned`, `XGBoost_Pipeline_v1`), ისე ოპტიმალურ run-ებს (`XGBoost_Cleaning`, `XGBoost_Feature_Engineering`, `XGBoost_Feature_Engineering_Aggregations`, `XGBoost_Feature_Selection_v1`, `XGBoost_Feature_Selection_Importance`, `XGBoost_Training_Baseline`, `XGBoost_Training_CrossValidation`, `XGBoost_Training_Optuna` + 25 nested trials, `XGBoost_Training_Final`, `XGBoost_Pipeline_Final`).

### ჩაწერილი მეტრიკების აღწერა

თითოეული run-ის შიგნით ფიქსირდება:

**Cleaning run-ები** (`*_Cleaning_*`):
- `memory_before_MB`, `memory_after_MB` — RAM ეფექტი
- `num_features`, `num_samples` — ზომები
- `nan_ratio_mean` — საშუალო missing ratio
- პარამეტრები: `strategy`, `nan_handling`

**Feature Engineering run-ები** (`*_FeatureEngineering_*`):
- `num_features_after_fe`, `num_new_features`
- პარამეტრები: `time_features`, `email_features`, `device_features`, `uid_features`

**Feature Selection run-ები** (`*_FeatureSelection_*`):
- `num_features_after_prune`, `num_features_kept_by_importance`
- `quick_val_auc` — სწრაფი XGBoost-ის შედეგი feature selection-ის ვერიფიკაციისთვის

**Training run-ები** (`*_Training_*`):
- `train_auc`, `val_auc` — ROC AUC-ის ფიქსაცია
- `best_iteration` — early stopping-ის შედეგი
- ყველა XGBoost hyperparameter

**Cross-Validation** (`XGBoost_Training_CrossValidation`):
- `fold_0_auc`, `fold_1_auc`, `fold_2_auc`, `cv_mean_auc`, `cv_std_auc`

**Optuna** (`XGBoost_Training_Optuna` + 25 nested):
- მშობელი run: `best_val_auc`, `n_trials`, `best_<param>`
- თითოეული trial: ყველა შერჩეული პარამეტრი + `val_auc` + `best_iteration`

**Final Pipeline** (`XGBoost_Pipeline_Final`):
- `train_auc`, `val_auc` — time-based holdout-ზე
- ყველა საბოლოო XGB hyperparameter (`xgb_*`)
- `pipeline_steps` — Pipeline-ის სტრუქტურა
- შენახული model artifact-ი signature-ით და input_example-ით

### საუკეთესო მოდელის შედეგები

**`XGBoost_Pipeline_Final`** (Model Registry: `XGBoost_FraudDetection` v1):

| მეტრიკა | მნიშვნელობა |
|---|---|
| Train AUC (time-based, train portion) | 0.9976 |
| Val AUC (time-based holdout) | 0.9348 |
| **Kaggle Public LB** | **0.9466** |
| **Kaggle Private LB** | **0.9163** |
| Pipeline steps | `feature_engineering → aggregations → encoding → pruner → model` |
| Final n_estimators | Optuna-სა და early stopping-ით შერჩეული |
| Registered as | `XGBoost_FraudDetection` (DagsHub MLflow Model Registry) |

საბაზისო `XGBoost_FraudDetection_v1`-თან შედარებამ გამოაჩინა, რომ ოპტიმიზაციამ რეალურ Leaderboard-ზე გააუმჯობესა შედეგი — და სწორედ ეს ადასტურებს ოპტიმალურ XGBoost-ში გატარებული feature engineering-ის ღირებულებას.