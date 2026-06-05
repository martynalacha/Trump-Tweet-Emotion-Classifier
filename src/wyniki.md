Używane urządzenie: cuda
Liczba klas: 11 (angry, fearful, joyful, sad, disgusted, surprised, enthusiastic, trusting, sarcastic, patriotic, neutral)

[1/8] Ładuję dataset (DistilBERT embeddingi, etykiety NRC)...
Pobieram dataset z Kaggle...
Using Colab cache for faster access to the 'trump-tweets' dataset.
Liczba tweetów: 43352

Generuję etykiety emocji (NRC + reguły)...
Etykietowanie: 100%|██████████| 43352/43352 [00:17<00:00, 2430.66it/s]

Rozkład etykiet:
 0. angry             6018 (13.88%)
 1. fearful           2991 (6.90%)
 2. joyful            6158 (14.20%)
 3. sad               1770 (4.08%)
 4. disgusted          451 (1.04%)
 5. surprised          586 (1.35%)
 6. enthusiastic      3738 (8.62%)
 7. trusting          5306 (12.24%)
 8. sarcastic         1258 (2.90%)
 9. patriotic         1833 (4.23%)
10. neutral          13243 (30.55%)

Ładuję model DistilBERT i tokenizer...
Loading weights: 100%
 100/100 [00:00<00:00, 489.51it/s, Materializing param=transformer.layer.5.sa_layer_norm.weight]
DistilBertModel LOAD REPORT from: distilbert-base-uncased
Key                     | Status     |  | 
------------------------+------------+--+-
vocab_transform.weight  | UNEXPECTED |  | 
vocab_layer_norm.bias   | UNEXPECTED |  | 
vocab_projector.bias    | UNEXPECTED |  | 
vocab_transform.bias    | UNEXPECTED |  | 
vocab_layer_norm.weight | UNEXPECTED |  | 

Notes:
- UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.
Generuję embeddingi DistilBERT (CLS token)...
DistilBERT: 100%|██████████| 678/678 [01:09<00:00,  9.71it/s]
Zapisano DistilBERT embeddingi: torch.Size([43352, 768])

Ładuję model SBERT: all-MiniLM-L6-v2...
Loading weights: 100%
 103/103 [00:00<00:00, 401.97it/s, Materializing param=pooler.dense.weight]
BertModel LOAD REPORT from: sentence-transformers/all-MiniLM-L6-v2
Key                     | Status     |  | 
------------------------+------------+--+-
embeddings.position_ids | UNEXPECTED |  | 

Notes:
- UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.
SBERT: 100%|██████████| 678/678 [00:23<00:00, 29.01it/s]
Zapisano SBERT embeddingi: torch.Size([43352, 384])

Pobieram / Ładuję model GloVe: glove-twitter-200...
Macierz GloVe: torch.Size([1193516, 200])
Zapisano GloVe dane. Vocab: 1193516, Sekwencje: 43352

Wszystkie potoki danych wejściowych zostały przygotowane pomyślnie!
  Rozkład etykiet:
    angry           6018
    fearful         2991
    joyful          6158
    sad             1770
    disgusted       451
    surprised       586
    enthusiastic    3738
    trusting        5306
    sarcastic       1258
    patriotic       1833
    neutral         13243

[2/8] Ładuję dataset (SBERT embeddingi)...
Pobieram dataset z Kaggle...
Using Colab cache for faster access to the 'trump-tweets' dataset.
Liczba tweetów: 43352

Generuję etykiety emocji (NRC + reguły)...
Etykietowanie: 100%|██████████| 43352/43352 [00:16<00:00, 2638.93it/s]

Rozkład etykiet:
 0. angry             6018 (13.88%)
 1. fearful           2991 (6.90%)
 2. joyful            6158 (14.20%)
 3. sad               1770 (4.08%)
 4. disgusted          451 (1.04%)
 5. surprised          586 (1.35%)
 6. enthusiastic      3738 (8.62%)
 7. trusting          5306 (12.24%)
 8. sarcastic         1258 (2.90%)
 9. patriotic         1833 (4.23%)
10. neutral          13243 (30.55%)

Ładuję model DistilBERT i tokenizer...
Loading weights: 100%
 100/100 [00:00<00:00, 418.48it/s, Materializing param=transformer.layer.5.sa_layer_norm.weight]
DistilBertModel LOAD REPORT from: distilbert-base-uncased
Key                     | Status     |  | 
------------------------+------------+--+-
vocab_transform.weight  | UNEXPECTED |  | 
vocab_layer_norm.bias   | UNEXPECTED |  | 
vocab_projector.bias    | UNEXPECTED |  | 
vocab_transform.bias    | UNEXPECTED |  | 
vocab_layer_norm.weight | UNEXPECTED |  | 

Notes:
- UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.
Generuję embeddingi DistilBERT (CLS token)...
DistilBERT: 100%|██████████| 678/678 [01:09<00:00,  9.73it/s]
Zapisano DistilBERT embeddingi: torch.Size([43352, 768])

Ładuję model SBERT: all-MiniLM-L6-v2...
Loading weights: 100%
 103/103 [00:00<00:00, 327.69it/s, Materializing param=pooler.dense.weight]
BertModel LOAD REPORT from: sentence-transformers/all-MiniLM-L6-v2
Key                     | Status     |  | 
------------------------+------------+--+-
embeddings.position_ids | UNEXPECTED |  | 

Notes:
- UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.
SBERT: 100%|██████████| 678/678 [00:22<00:00, 29.94it/s]
Zapisano SBERT embeddingi: torch.Size([43352, 384])

Pobieram / Ładuję model GloVe: glove-twitter-200...
Macierz GloVe: torch.Size([1193516, 200])
Zapisano GloVe dane. Vocab: 1193516, Sekwencje: 43352

Wszystkie potoki danych wejściowych zostały przygotowane pomyślnie!

[3/8] Ładuję dataset (GloVe + tokenizacja)...
Pobieram dataset z Kaggle...
Using Colab cache for faster access to the 'trump-tweets' dataset.
Liczba tweetów: 43352

Generuję etykiety emocji (NRC + reguły)...
Etykietowanie: 100%|██████████| 43352/43352 [00:17<00:00, 2516.87it/s]

Rozkład etykiet:
 0. angry             6018 (13.88%)
 1. fearful           2991 (6.90%)
 2. joyful            6158 (14.20%)
 3. sad               1770 (4.08%)
 4. disgusted          451 (1.04%)
 5. surprised          586 (1.35%)
 6. enthusiastic      3738 (8.62%)
 7. trusting          5306 (12.24%)
 8. sarcastic         1258 (2.90%)
 9. patriotic         1833 (4.23%)
10. neutral          13243 (30.55%)

Ładuję model DistilBERT i tokenizer...
Loading weights: 100%
 100/100 [00:00<00:00, 510.18it/s, Materializing param=transformer.layer.5.sa_layer_norm.weight]
DistilBertModel LOAD REPORT from: distilbert-base-uncased
Key                     | Status     |  | 
------------------------+------------+--+-
vocab_transform.weight  | UNEXPECTED |  | 
vocab_layer_norm.bias   | UNEXPECTED |  | 
vocab_projector.bias    | UNEXPECTED |  | 
vocab_transform.bias    | UNEXPECTED |  | 
vocab_layer_norm.weight | UNEXPECTED |  | 

Notes:
- UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.
Generuję embeddingi DistilBERT (CLS token)...
DistilBERT: 100%|██████████| 678/678 [01:09<00:00,  9.71it/s]
Zapisano DistilBERT embeddingi: torch.Size([43352, 768])

Ładuję model SBERT: all-MiniLM-L6-v2...
Loading weights: 100%
 103/103 [00:00<00:00, 475.11it/s, Materializing param=pooler.dense.weight]
BertModel LOAD REPORT from: sentence-transformers/all-MiniLM-L6-v2
Key                     | Status     |  | 
------------------------+------------+--+-
embeddings.position_ids | UNEXPECTED |  | 

Notes:
- UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.
SBERT: 100%|██████████| 678/678 [00:23<00:00, 29.44it/s]
Zapisano SBERT embeddingi: torch.Size([43352, 384])

Pobieram / Ładuję model GloVe: glove-twitter-200...
Macierz GloVe: torch.Size([1193516, 200])
Zapisano GloVe dane. Vocab: 1193516, Sekwencje: 43352

Wszystkie potoki danych wejściowych zostały przygotowane pomyślnie!

[4/8] === PODEJŚCIE 1: DistilBERT + MLP ===

==================================================
Trening modelu: distilbert_mlp
==================================================
Epoka   1/30 | train_loss=2.3251 train_acc=0.1961 | val_loss=2.2251 val_acc=0.3428
Epoka   2/30 | train_loss=2.2193 train_acc=0.2386 | val_loss=2.1738 val_acc=0.2093
Epoka   3/30 | train_loss=2.1707 train_acc=0.2430 | val_loss=2.1566 val_acc=0.2744
Epoka   4/30 | train_loss=2.1570 train_acc=0.2527 | val_loss=2.1400 val_acc=0.2891
Epoka   5/30 | train_loss=2.1416 train_acc=0.2484 | val_loss=2.1759 val_acc=0.2910
Epoka   6/30 | train_loss=2.1351 train_acc=0.2518 | val_loss=2.1271 val_acc=0.2725
Epoka   7/30 | train_loss=2.1173 train_acc=0.2514 | val_loss=2.0962 val_acc=0.2313
Epoka   8/30 | train_loss=2.1128 train_acc=0.2570 | val_loss=2.1657 val_acc=0.2731
Epoka   9/30 | train_loss=2.0945 train_acc=0.2578 | val_loss=2.1588 val_acc=0.2894
Epoka  10/30 | train_loss=2.0955 train_acc=0.2649 | val_loss=2.0865 val_acc=0.2878
Epoka  11/30 | train_loss=2.0774 train_acc=0.2615 | val_loss=2.0734 val_acc=0.2662
Epoka  12/30 | train_loss=2.0723 train_acc=0.2605 | val_loss=2.1511 val_acc=0.2971
Epoka  13/30 | train_loss=2.0637 train_acc=0.2659 | val_loss=2.1063 val_acc=0.2795
Epoka  14/30 | train_loss=2.0560 train_acc=0.2656 | val_loss=2.0956 val_acc=0.2804
Epoka  15/30 | train_loss=2.0549 train_acc=0.2649 | val_loss=2.1380 val_acc=0.2547
Epoka  16/30 | train_loss=2.0088 train_acc=0.2776 | val_loss=2.0880 val_acc=0.2525

Early stopping po epoce 16.

Najlepszy model zapisany: models/distilbert_mlp_best.pt
Najlepsza val_loss: 2.0734
Wykres zapisany: plots/history_distilbert.png


============================================================
Wyniki: DistilBERT_MLP
============================================================
              precision    recall  f1-score   support

       angry       0.35      0.05      0.09       906
     fearful       0.17      0.26      0.21       455
      joyful       0.37      0.35      0.36       928
         sad       0.10      0.37      0.16       263
   disgusted       0.03      0.20      0.05        56
   surprised       0.05      0.29      0.09       101
enthusiastic       0.18      0.06      0.09       556
    trusting       0.17      0.04      0.06       803
   sarcastic       0.13      0.33      0.19       200
   patriotic       0.18      0.65      0.28       263
     neutral       0.80      0.42      0.55      1971

    accuracy                           0.27      6502
   macro avg       0.23      0.27      0.19      6502
weighted avg       0.41      0.27      0.29      6502

Confusion matrix zapisana: plots/confusion_matrix_DistilBERT_MLP.png


[5/8] === PODEJŚCIE 2: SBERT + MLP ===

==================================================
Trening modelu: sbert_mlp
==================================================
Epoka   1/30 | train_loss=2.2110 train_acc=0.2451 | val_loss=2.0692 val_acc=0.2353
Epoka   2/30 | train_loss=2.0743 train_acc=0.2811 | val_loss=2.0020 val_acc=0.2996
Epoka   3/30 | train_loss=2.0075 train_acc=0.2914 | val_loss=2.0235 val_acc=0.2845
Epoka   4/30 | train_loss=1.9612 train_acc=0.3038 | val_loss=1.9978 val_acc=0.3050
Epoka   5/30 | train_loss=1.9240 train_acc=0.3125 | val_loss=2.0165 val_acc=0.3339
Epoka   6/30 | train_loss=1.8924 train_acc=0.3189 | val_loss=2.0008 val_acc=0.2890
Epoka   7/30 | train_loss=1.8530 train_acc=0.3274 | val_loss=2.0600 val_acc=0.3353
Epoka   8/30 | train_loss=1.8297 train_acc=0.3307 | val_loss=2.0308 val_acc=0.3107
Epoka   9/30 | train_loss=1.7300 train_acc=0.3449 | val_loss=2.1177 val_acc=0.3654

Early stopping po epoce 9.

Najlepszy model zapisany: models/sbert_mlp_best.pt
Najlepsza val_loss: 1.9978
Wykres zapisany: plots/history_sbert.png


============================================================
Wyniki: SBERT_MLP
============================================================
              precision    recall  f1-score   support

       angry       0.36      0.07      0.11       906
     fearful       0.19      0.36      0.25       455
      joyful       0.36      0.48      0.41       928
         sad       0.13      0.48      0.21       263
   disgusted       0.06      0.43      0.10        56
   surprised       0.06      0.04      0.05       101
enthusiastic       0.30      0.30      0.30       556
    trusting       0.36      0.20      0.26       803
   sarcastic       0.14      0.27      0.18       200
   patriotic       0.25      0.56      0.35       263
     neutral       0.84      0.36      0.50      1971

    accuracy                           0.32      6502
   macro avg       0.28      0.32      0.25      6502
weighted avg       0.46      0.32      0.33      6502

Confusion matrix zapisana: plots/confusion_matrix_SBERT_MLP.png


[6/8] === PODEJŚCIE 3: GloVe + BiLSTM ===

==================================================
Trening modelu: glove_bilstm
==================================================
Epoka   1/30 | train_loss=2.1780 train_acc=0.2638 | val_loss=2.0353 val_acc=0.3605
Epoka   2/30 | train_loss=1.9492 train_acc=0.3598 | val_loss=1.8713 val_acc=0.4086
Epoka   3/30 | train_loss=1.8157 train_acc=0.4090 | val_loss=1.7739 val_acc=0.4434
Epoka   4/30 | train_loss=1.7123 train_acc=0.4385 | val_loss=1.7604 val_acc=0.4491
Epoka   5/30 | train_loss=1.6391 train_acc=0.4600 | val_loss=1.6374 val_acc=0.4920
Epoka   6/30 | train_loss=1.5731 train_acc=0.4794 | val_loss=1.6347 val_acc=0.4725
Epoka   7/30 | train_loss=1.4979 train_acc=0.4971 | val_loss=1.6541 val_acc=0.5197
Epoka   8/30 | train_loss=1.4384 train_acc=0.5114 | val_loss=1.6483 val_acc=0.4906
Epoka   9/30 | train_loss=1.3942 train_acc=0.5251 | val_loss=1.6244 val_acc=0.4905
Epoka  10/30 | train_loss=1.3436 train_acc=0.5335 | val_loss=1.5780 val_acc=0.5023
Epoka  11/30 | train_loss=1.2991 train_acc=0.5430 | val_loss=1.6245 val_acc=0.4983
Epoka  12/30 | train_loss=1.2534 train_acc=0.5556 | val_loss=1.6390 val_acc=0.5052
Epoka  13/30 | train_loss=1.2168 train_acc=0.5625 | val_loss=1.7066 val_acc=0.5328
Epoka  14/30 | train_loss=1.1690 train_acc=0.5723 | val_loss=1.7130 val_acc=0.5174
Epoka  15/30 | train_loss=1.0462 train_acc=0.5944 | val_loss=1.7599 val_acc=0.5417

Early stopping po epoce 15.

Najlepszy model zapisany: models/glove_bilstm_best.pt
Najlepsza val_loss: 1.5780
Wykres zapisany: plots/history_glove.png


============================================================
Wyniki: GloVe_BiLSTM
============================================================
              precision    recall  f1-score   support

       angry       0.64      0.33      0.43       906
     fearful       0.46      0.47      0.47       455
      joyful       0.56      0.53      0.55       928
         sad       0.44      0.53      0.48       263
   disgusted       0.11      0.57      0.18        56
   surprised       0.20      0.49      0.28       101
enthusiastic       0.50      0.56      0.53       556
    trusting       0.55      0.32      0.41       803
   sarcastic       0.19      0.54      0.28       200
   patriotic       0.33      0.72      0.45       263
     neutral       0.76      0.61      0.68      1971

    accuracy                           0.51      6502
   macro avg       0.43      0.52      0.43      6502
weighted avg       0.59      0.51      0.53      6502

Confusion matrix zapisana: plots/confusion_matrix_GloVe_BiLSTM.png


[7/8] === PODEJŚCIE 4: Zero-Shot BART (11 etykiet) ===
Loading zero-shot model: facebook/bart-large-mnli...
config.json: 100%
 1.15k/1.15k [00:00<00:00, 55.4kB/s]
model.safetensors: 100%
 1.63G/1.63G [00:12<00:00, 73.8MB/s]
Loading weights: 100%
 515/515 [00:00<00:00, 803.39it/s, Materializing param=model.shared.weight]
tokenizer_config.json: 100%
 26.0/26.0 [00:00<00:00, 1.45kB/s]
vocab.json: 100%
 899k/899k [00:00<00:00, 15.0MB/s]
merges.txt: 100%
 456k/456k [00:00<00:00, 10.2MB/s]
tokenizer.json: 100%
 1.36M/1.36M [00:00<00:00, 14.4MB/s]
Classifying 2000 tweets using native GPU batching...
100%|██████████| 2000/2000 [00:00<00:00, 64373.75it/s]
Results saved to: data/processed/zero_shot_results.json

============================================================
Wyniki Zero-Shot (BART) — 11 klas emocji
============================================================
              precision    recall  f1-score   support

       angry       0.22      0.31      0.25       265
     fearful       0.28      0.19      0.23       178
      joyful       0.38      0.06      0.10       260
         sad       0.15      0.10      0.12       145
   disgusted       0.01      0.23      0.03        13
   surprised       0.03      0.38      0.06        29
enthusiastic       0.13      0.20      0.16       153
    trusting       0.15      0.08      0.10       216
   sarcastic       0.02      0.22      0.03        18
   patriotic       0.12      0.21      0.15        73
     neutral       0.62      0.06      0.11       650

    accuracy                           0.13      2000
   macro avg       0.19      0.18      0.12      2000
weighted avg       0.34      0.13      0.14      2000

Confusion matrix zapisana: plots/confusion_matrix_ZeroShot.png

Rozkład etykiet zapisany: plots/zero_shot_label_distribution.png

Rozkład pewności zapisany: plots/zero_shot_confidence.png

Przykłady zapisane: plots/zero_shot_examples.png


[8/8] Porównanie wszystkich modeli...
Porównanie modeli zapisane: plots/model_comparison.png


Analiza SHAP (Podejście 1 i 2)...
Obliczam SHAP (DeepExplainer) dla: DistilBERT_MLP...
SHAP obliczone. Liczba klas: 11, kształt (klasa 0): (100, 768)
/content/src/shap_analysis.py:119: FutureWarning: The NumPy global RNG was seeded by calling `np.random.seed`. In a future version this function will no longer use the global RNG. Pass `rng` explicitly to opt-in to the new behaviour and silence this warning.
  shap.summary_plot(
SHAP summary zapisany: plots/shap_summary_DistilBERT_MLP_class0.png

/content/src/shap_analysis.py:119: FutureWarning: The NumPy global RNG was seeded by calling `np.random.seed`. In a future version this function will no longer use the global RNG. Pass `rng` explicitly to opt-in to the new behaviour and silence this warning.
  shap.summary_plot(
SHAP summary zapisany: plots/shap_summary_DistilBERT_MLP_class1.png

/content/src/shap_analysis.py:119: FutureWarning: The NumPy global RNG was seeded by calling `np.random.seed`. In a future version this function will no longer use the global RNG. Pass `rng` explicitly to opt-in to the new behaviour and silence this warning.
  shap.summary_plot(
SHAP summary zapisany: plots/shap_summary_DistilBERT_MLP_class2.png

/content/src/shap_analysis.py:119: FutureWarning: The NumPy global RNG was seeded by calling `np.random.seed`. In a future version this function will no longer use the global RNG. Pass `rng` explicitly to opt-in to the new behaviour and silence this warning.
  shap.summary_plot(
SHAP summary zapisany: plots/shap_summary_DistilBERT_MLP_class3.png

/content/src/shap_analysis.py:119: FutureWarning: The NumPy global RNG was seeded by calling `np.random.seed`. In a future version this function will no longer use the global RNG. Pass `rng` explicitly to opt-in to the new behaviour and silence this warning.
  shap.summary_plot(
SHAP summary zapisany: plots/shap_summary_DistilBERT_MLP_class4.png

/content/src/shap_analysis.py:119: FutureWarning: The NumPy global RNG was seeded by calling `np.random.seed`. In a future version this function will no longer use the global RNG. Pass `rng` explicitly to opt-in to the new behaviour and silence this warning.
  shap.summary_plot(
SHAP summary zapisany: plots/shap_summary_DistilBERT_MLP_class5.png

/content/src/shap_analysis.py:119: FutureWarning: The NumPy global RNG was seeded by calling `np.random.seed`. In a future version this function will no longer use the global RNG. Pass `rng` explicitly to opt-in to the new behaviour and silence this warning.
  shap.summary_plot(
SHAP summary zapisany: plots/shap_summary_DistilBERT_MLP_class6.png

/content/src/shap_analysis.py:119: FutureWarning: The NumPy global RNG was seeded by calling `np.random.seed`. In a future version this function will no longer use the global RNG. Pass `rng` explicitly to opt-in to the new behaviour and silence this warning.
  shap.summary_plot(
SHAP summary zapisany: plots/shap_summary_DistilBERT_MLP_class7.png

/content/src/shap_analysis.py:119: FutureWarning: The NumPy global RNG was seeded by calling `np.random.seed`. In a future version this function will no longer use the global RNG. Pass `rng` explicitly to opt-in to the new behaviour and silence this warning.
  shap.summary_plot(
SHAP summary zapisany: plots/shap_summary_DistilBERT_MLP_class8.png

/content/src/shap_analysis.py:119: FutureWarning: The NumPy global RNG was seeded by calling `np.random.seed`. In a future version this function will no longer use the global RNG. Pass `rng` explicitly to opt-in to the new behaviour and silence this warning.
  shap.summary_plot(
SHAP summary zapisany: plots/shap_summary_DistilBERT_MLP_class9.png

/content/src/shap_analysis.py:119: FutureWarning: The NumPy global RNG was seeded by calling `np.random.seed`. In a future version this function will no longer use the global RNG. Pass `rng` explicitly to opt-in to the new behaviour and silence this warning.
  shap.summary_plot(
SHAP summary zapisany: plots/shap_summary_DistilBERT_MLP_class10.png

SHAP bar plot zapisany: plots/shap_bar_DistilBERT_MLP.png

Obliczam SHAP (DeepExplainer) dla: SBERT_MLP...
SHAP obliczone. Liczba klas: 11, kształt (klasa 0): (100, 384)
/content/src/shap_analysis.py:119: FutureWarning: The NumPy global RNG was seeded by calling `np.random.seed`. In a future version this function will no longer use the global RNG. Pass `rng` explicitly to opt-in to the new behaviour and silence this warning.
  shap.summary_plot(
SHAP summary zapisany: plots/shap_summary_SBERT_MLP_class0.png

/content/src/shap_analysis.py:119: FutureWarning: The NumPy global RNG was seeded by calling `np.random.seed`. In a future version this function will no longer use the global RNG. Pass `rng` explicitly to opt-in to the new behaviour and silence this warning.
  shap.summary_plot(
SHAP summary zapisany: plots/shap_summary_SBERT_MLP_class1.png

/content/src/shap_analysis.py:119: FutureWarning: The NumPy global RNG was seeded by calling `np.random.seed`. In a future version this function will no longer use the global RNG. Pass `rng` explicitly to opt-in to the new behaviour and silence this warning.
  shap.summary_plot(
SHAP summary zapisany: plots/shap_summary_SBERT_MLP_class2.png

/content/src/shap_analysis.py:119: FutureWarning: The NumPy global RNG was seeded by calling `np.random.seed`. In a future version this function will no longer use the global RNG. Pass `rng` explicitly to opt-in to the new behaviour and silence this warning.
  shap.summary_plot(
SHAP summary zapisany: plots/shap_summary_SBERT_MLP_class3.png

/content/src/shap_analysis.py:119: FutureWarning: The NumPy global RNG was seeded by calling `np.random.seed`. In a future version this function will no longer use the global RNG. Pass `rng` explicitly to opt-in to the new behaviour and silence this warning.
  shap.summary_plot(
SHAP summary zapisany: plots/shap_summary_SBERT_MLP_class4.png

/content/src/shap_analysis.py:119: FutureWarning: The NumPy global RNG was seeded by calling `np.random.seed`. In a future version this function will no longer use the global RNG. Pass `rng` explicitly to opt-in to the new behaviour and silence this warning.
  shap.summary_plot(
SHAP summary zapisany: plots/shap_summary_SBERT_MLP_class5.png

/content/src/shap_analysis.py:119: FutureWarning: The NumPy global RNG was seeded by calling `np.random.seed`. In a future version this function will no longer use the global RNG. Pass `rng` explicitly to opt-in to the new behaviour and silence this warning.
  shap.summary_plot(
SHAP summary zapisany: plots/shap_summary_SBERT_MLP_class6.png

/content/src/shap_analysis.py:119: FutureWarning: The NumPy global RNG was seeded by calling `np.random.seed`. In a future version this function will no longer use the global RNG. Pass `rng` explicitly to opt-in to the new behaviour and silence this warning.
  shap.summary_plot(
SHAP summary zapisany: plots/shap_summary_SBERT_MLP_class7.png

/content/src/shap_analysis.py:119: FutureWarning: The NumPy global RNG was seeded by calling `np.random.seed`. In a future version this function will no longer use the global RNG. Pass `rng` explicitly to opt-in to the new behaviour and silence this warning.
  shap.summary_plot(
SHAP summary zapisany: plots/shap_summary_SBERT_MLP_class8.png

/content/src/shap_analysis.py:119: FutureWarning: The NumPy global RNG was seeded by calling `np.random.seed`. In a future version this function will no longer use the global RNG. Pass `rng` explicitly to opt-in to the new behaviour and silence this warning.
  shap.summary_plot(
SHAP summary zapisany: plots/shap_summary_SBERT_MLP_class9.png

/content/src/shap_analysis.py:119: FutureWarning: The NumPy global RNG was seeded by calling `np.random.seed`. In a future version this function will no longer use the global RNG. Pass `rng` explicitly to opt-in to the new behaviour and silence this warning.
  shap.summary_plot(
SHAP summary zapisany: plots/shap_summary_SBERT_MLP_class10.png

SHAP bar plot zapisany: plots/shap_bar_SBERT_MLP.png

Trening baseline TF-IDF + LogReg...
/usr/local/lib/python3.12/dist-packages/sklearn/linear_model/_logistic.py:1247: FutureWarning: 'multi_class' was deprecated in version 1.5 and will be removed in 1.7. From then on, it will always use 'multinomial'. Leave it to its default value to avoid this warning.
  warnings.warn(

Wyniki baseline TF-IDF + LogReg:
              precision    recall  f1-score   support

       angry       0.58      0.66      0.61      1204
     fearful       0.67      0.46      0.54       598
      joyful       0.67      0.63      0.65      1232
         sad       0.74      0.39      0.51       354
   disgusted       0.83      0.11      0.20        90
   surprised       0.64      0.06      0.11       117
enthusiastic       0.70      0.64      0.67       748
    trusting       0.61      0.58      0.59      1061
   sarcastic       0.51      0.11      0.18       251
   patriotic       0.59      0.44      0.51       367
     neutral       0.67      0.88      0.76      2649

    accuracy                           0.65      8671
   macro avg       0.65      0.45      0.48      8671
weighted avg       0.65      0.65      0.63      8671

Obliczam SHAP (LinearExplainer)...
/usr/local/lib/python3.12/dist-packages/shap/explainers/_linear.py:123: FutureWarning: The feature_perturbation option is now deprecated in favor of using the appropriate masker (maskers.Independent, maskers.Partition or maskers.Impute).
  warnings.warn(wmsg, FutureWarning)
/content/src/shap_analysis.py:222: FutureWarning: The NumPy global RNG was seeded by calling `np.random.seed`. In a future version this function will no longer use the global RNG. Pass `rng` explicitly to opt-in to the new behaviour and silence this warning.
  shap.summary_plot(
SHAP TF-IDF zapisany: plots/shap_tfidf_baseline_class0.png

/content/src/shap_analysis.py:222: FutureWarning: The NumPy global RNG was seeded by calling `np.random.seed`. In a future version this function will no longer use the global RNG. Pass `rng` explicitly to opt-in to the new behaviour and silence this warning.
  shap.summary_plot(
SHAP TF-IDF zapisany: plots/shap_tfidf_baseline_class1.png

/content/src/shap_analysis.py:222: FutureWarning: The NumPy global RNG was seeded by calling `np.random.seed`. In a future version this function will no longer use the global RNG. Pass `rng` explicitly to opt-in to the new behaviour and silence this warning.
  shap.summary_plot(
SHAP TF-IDF zapisany: plots/shap_tfidf_baseline_class2.png

/content/src/shap_analysis.py:222: FutureWarning: The NumPy global RNG was seeded by calling `np.random.seed`. In a future version this function will no longer use the global RNG. Pass `rng` explicitly to opt-in to the new behaviour and silence this warning.
  shap.summary_plot(
SHAP TF-IDF zapisany: plots/shap_tfidf_baseline_class3.png

/content/src/shap_analysis.py:222: FutureWarning: The NumPy global RNG was seeded by calling `np.random.seed`. In a future version this function will no longer use the global RNG. Pass `rng` explicitly to opt-in to the new behaviour and silence this warning.
  shap.summary_plot(
SHAP TF-IDF zapisany: plots/shap_tfidf_baseline_class4.png

/content/src/shap_analysis.py:222: FutureWarning: The NumPy global RNG was seeded by calling `np.random.seed`. In a future version this function will no longer use the global RNG. Pass `rng` explicitly to opt-in to the new behaviour and silence this warning.
  shap.summary_plot(
SHAP TF-IDF zapisany: plots/shap_tfidf_baseline_class5.png

/content/src/shap_analysis.py:222: FutureWarning: The NumPy global RNG was seeded by calling `np.random.seed`. In a future version this function will no longer use the global RNG. Pass `rng` explicitly to opt-in to the new behaviour and silence this warning.
  shap.summary_plot(
SHAP TF-IDF zapisany: plots/shap_tfidf_baseline_class6.png

/content/src/shap_analysis.py:222: FutureWarning: The NumPy global RNG was seeded by calling `np.random.seed`. In a future version this function will no longer use the global RNG. Pass `rng` explicitly to opt-in to the new behaviour and silence this warning.
  shap.summary_plot(
SHAP TF-IDF zapisany: plots/shap_tfidf_baseline_class7.png

/content/src/shap_analysis.py:222: FutureWarning: The NumPy global RNG was seeded by calling `np.random.seed`. In a future version this function will no longer use the global RNG. Pass `rng` explicitly to opt-in to the new behaviour and silence this warning.
  shap.summary_plot(
SHAP TF-IDF zapisany: plots/shap_tfidf_baseline_class8.png

/content/src/shap_analysis.py:222: FutureWarning: The NumPy global RNG was seeded by calling `np.random.seed`. In a future version this function will no longer use the global RNG. Pass `rng` explicitly to opt-in to the new behaviour and silence this warning.
  shap.summary_plot(
SHAP TF-IDF zapisany: plots/shap_tfidf_baseline_class9.png

/content/src/shap_analysis.py:222: FutureWarning: The NumPy global RNG was seeded by calling `np.random.seed`. In a future version this function will no longer use the global RNG. Pass `rng` explicitly to opt-in to the new behaviour and silence this warning.
  shap.summary_plot(
SHAP TF-IDF zapisany: plots/shap_tfidf_baseline_class10.png


============================================================
PODSUMOWANIE WYNIKÓW (11 klas emocji)
============================================================
  Podejście 1 (DistilBERT+MLP): acc=0.270  f1_macro=0.193
  Podejście 2 (SBERT+MLP):      acc=0.316  f1_macro=0.247
  Podejście 3 (GloVe+BiLSTM):   acc=0.509  f1_macro=0.431
  Podejście 4 (Zero-Shot BART):  acc=0.132  f1_macro=0.121

✓ Wszystkie wyniki zapisane w 'plots/'.
<Figure size 640x480 with 0 Axes>




Używane urządzenie: cuda
Hiperparametry: samples=8000, epochs=3, lr_bert=2e-05, lr_head=0.001, batch=32

[1/4] Ładuję dataset (tokeny DistilBERT, podzbiór stratyfikowany)...
Komplet wszystkich danych (etykiety + 3 rodzaje embeddingów) już istnieje na dysku.
Ładuję tokenizer DistilBERT: distilbert-base-uncased
Tokenizuję 8000 tweetów (max_length=64)...
  input_ids shape: torch.Size([8000, 64])
  Rozkład klas (podzbiór): [1111, 552, 1136, 327, 83, 108, 690, 979, 232, 338, 2444]
  Train: 5600 | Val: 1200 | Test: 1200

[2/4] Inicjalizuję model DistilBERT + klasyfikator...
Loading weights: 100%
 100/100 [00:00<00:00, 173.41it/s, Materializing param=transformer.layer.5.sa_layer_norm.weight]
DistilBertModel LOAD REPORT from: distilbert-base-uncased
Key                     | Status     |  | 
------------------------+------------+--+-
vocab_transform.weight  | UNEXPECTED |  | 
vocab_layer_norm.bias   | UNEXPECTED |  | 
vocab_projector.bias    | UNEXPECTED |  | 
vocab_transform.bias    | UNEXPECTED |  | 
vocab_layer_norm.weight | UNEXPECTED |  | 

Notes:
- UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.

[3/4] Trening (3 epoki, early stopping patience=2)...
============================================================
Epoka  1/3 | train_loss=2.4005 train_acc=0.0900 | val_loss=2.3985 val_acc=0.1000
Epoka  2/3 | train_loss=2.3911 train_acc=0.1286 | val_loss=2.3917 val_acc=0.1775
Epoka  3/3 | train_loss=2.3719 train_acc=0.2245 | val_loss=2.3699 val_acc=0.3100

Najlepszy model zapisany: models/distilbert_ft_best.pt
Najlepsza val_loss: 2.3699

[4/4] Ewaluacja na zbiorze testowym...
Wykres treningowy zapisany: plots/history_distilbert_ft.png


============================================================
Wyniki: DistilBERT Fine-tuned (end-to-end)
============================================================
              precision    recall  f1-score   support

       angry       0.33      0.47      0.39       185
     fearful       0.08      0.04      0.05        72
      joyful       0.26      0.35      0.30       149
         sad       0.10      0.18      0.13        51
   disgusted       0.00      0.00      0.00        13
   surprised       0.00      0.00      0.00        17
enthusiastic       0.40      0.02      0.03       116
    trusting       0.17      0.02      0.03       126
   sarcastic       0.00      0.00      0.00        33
   patriotic       0.18      0.43      0.25        54
     neutral       0.51      0.62      0.56       384

    accuracy                           0.35      1200
   macro avg       0.18      0.19      0.16      1200
weighted avg       0.32      0.35      0.30      1200

Confusion matrix zapisana: plots/confusion_matrix_DistilBERT_FT.png


============================================================
PODSUMOWANIE — DistilBERT Fine-tuned vs Frozen
============================================================
  DistilBERT FT  (end-to-end): acc=0.346  f1_macro=0.158
  DistilBERT MLP (zamrożony):  acc=0.297  f1_macro=0.215  [z main.py]
  GloVe+BiLSTM   (referencja): acc=0.481  f1_macro=0.412  [z main.py]

✓ Wykresy zapisane w 'plots/'.
✓ Model zapisany: models/distilbert_ft_best.pt
<Figure size 640x480 with 0 Axes> 