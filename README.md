# Product Dimension Prediction

Solution for the olympiad regression competition: predicting physical product dimensions (weight, height, length, width) from product listings with text, tabular metadata, and images.

**Best submission: `submission_vision_blend.csv`** (Vision-Text Blend)

---

## Solution Overview

### Pipeline

```
Raw Data (text + images + tabular)
    |
    v
Feature Engineering (features.py, features2.py, text_features.py)
    |
    v
Embedding Extraction (embeddings.py, emb2.py, image_embeddings.py, image_vit_embeddings.py)
    |
    v
Dimensionality Reduction (pca_emb.py) + KNN Features (knn_feats.py) + Aggregates (build_agg.py)
    |
    v
Model Training (CatBoost, LightGBM, Neural Network, Linear)
    |
    v
Blending (blend2.py, make_vision_blend.py)
    |
    v
Post-processing (make_sub.py)
```

### Features

| Category | Description |
|----------|-------------|
| **Tabular** | Category, subcategory, microcategory, seller/buyer IDs, price, item condition, date features |
| **Text** | Title + description TF-IDF (word/char n-grams) -> SVD 320d, multilingual-e5 embeddings (384d + 1024d PCA 192d) |
| **Image** | ResNet50 (PCA 192d) + ViT-B/16 (PCA 192d) embeddings |
| **Engineered** | Regex unit extraction (kg, cm, ml, etc.), dimension triple parsing, keyword groups, fold-aware target encoding aggregates, KNN similarity features (k=8,32) |
| **Total** | ~860 features for CatBoost, sparse TF-IDF for LightGBM |

### Models

| Model | Features | Weight MAE | Height MAE | Length MAE | Width MAE | Score |
|-------|----------|-----------|-----------|-----------|----------|-------|
| Baseline CatBoost | 380 tabular | 0.2580 | 0.4907 | 0.2834 | 0.2919 | 0.752 |
| + Text SVD | +320 SVD | 0.2504 | 0.4876 | 0.2822 | 0.2905 | 0.754 |
| cat_v2 | +emb+extra+agg (831) | 0.2416 | 0.4828 | 0.2797 | 0.2882 | 0.756 |
| **cat_v3** | +KNN (859) | **0.2317** | **0.4772** | **0.2739** | **0.2836** | **0.760** |
| LightGBM | sparse (25K) | 0.2513 | 0.4923 | 0.2837 | 0.2916 | 0.752 |
| Neural Net v2 | embeddings+SVD+KNN | - | - | - | - | 0.756 |
| **Vision Blend** | cat_v3 + image models | - | - | - | - | **~0.77** |

### Blending Strategy

The final submission blends **cat_v3** (CatBoost with all features) with per-target **image+text CatBoost models**:

```python
prediction = (1 - w) * cat_v3_prediction + w * image_text_prediction
```

Per-target image weights: `[0.43039, 0.49391, 0.48621, 0.48355]` (weight, height, length, width)

### Post-processing

1. `expm1` inverse transform (targets trained in log1p space)
2. Weight clipped to `[0.001, 5000]`
3. Dimensions clipped to minimum `1.0`
4. **Ordered dimensions enforced**: height <= width <= length
5. Dimensions rounded to nearest integer

---

## Project Structure

```
.
├── common2.py                  # Shared utilities, metrics, fold generation
├── features.py                 # Base feature engineering (tabular + regex)
├── features2.py                # Extended features + aggregates
├── text_features.py            # TF-IDF + SVD text features
├── embeddings.py               # e5-small text embeddings
├── emb2.py                     # e5-large text embeddings
├── image_embeddings.py         # ResNet50 image embeddings
├── image_vit_embeddings.py     # ViT-B/16 image embeddings
├── pca_emb.py                  # PCA dimensionality reduction
├── knn_feats.py                # KNN similarity features
├── build_agg.py                # Aggregate feature builder
├── aggregate_features.py       # Earlier aggregate builder
│
├── cat2.py                     # CatBoost training (primary model)
├── lgb2.py                     # LightGBM training
├── nn2.py                      # Neural network training (PyTorch)
├── train_cat.py                # CatBoost training (text-native)
├── train_lgbm.py               # LightGBM training
├── train_linear.py             # Linear model
├── bert2.py                    # BERT fine-tuning (rubert-tiny2)
│
├── blend2.py                   # Multi-model optimal blending
├── make_vision_blend.py        # Final vision blend submission
├── make_sub.py                 # Post-processing utilities
│
├── submissions/
│   └── submission_vision_blend.csv  # Best submission
│
└── sample_submission.csv       # Sample format
```

---

## Reproduction

### Requirements

```bash
pip install -r requirements.txt
```

### Step 1: Feature Engineering

```bash
python features.py       # Base features -> artifacts/features.pkl
python features2.py      # Extended features -> artifacts/x_extra.pkl, agg5.npy
python text_features.py  # TF-IDF + SVD -> artifacts/text_svd.npy
python embeddings.py     # e5-small -> artifacts/embeddings.npy
python emb2.py           # e5-large -> artifacts/emb_e5large.npy
python pca_emb.py        # PCA reduction
python knn_feats.py      # KNN features -> artifacts/knn.npy
```

### Step 2: Image Embeddings

```bash
python image_embeddings.py     # ResNet50 -> artifacts/image_resnet50.npy
python image_vit_embeddings.py # ViT-B/16 -> artifacts/image_vit_b16.npy
```

### Step 3: Train Models

```bash
# CatBoost v3 (best single model)
python cat2.py --name cat_v3 --text --svd --emb --emb2 emb_e5large.npy --emb-pca --knn --knnfile knn.npy --agg --depth 8 --lr 0.045 --iters 5000 --stop 250

# LightGBM
python lgb2.py

# Neural Network
python nn2.py

# CatBoost with text (full train for final submission)
python train_cat.py --name final_text_full --text --svd --full --iters 1900
```

### Step 4: Blend & Submit

```bash
python blend2.py              # Optimize blend weights
python make_vision_blend.py   # Generate final submission
```

---

## Key Techniques

1. **Homoglyph-aware text normalization**: Latin/Cyrillic character normalization for mixed-script Russian marketplace titles
2. **Fold-aware target encoding**: 9 grouping keys with out-of-fold statistics to prevent leakage
3. **KNN features from embeddings**: Nearest neighbor target values as features (cosine similarity in e5-large space)
4. **Multi-scale text representation**: TF-IDF sparse + SVD dense + e5-small + e5-large embeddings
5. **Per-target vision blending**: Different blend weights per dimension (images are more informative for height/width)
6. **Physical constraint enforcement**: Ordered dimensions (h <= w <= l) in post-processing

---

## License

MIT
