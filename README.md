# Santander Product Recommendation — Per-Product Classifiers

Complementary analysis to the [LightFM hybrid repository](https://github.com/YOUR_USER/santander-lightfm).

Same competition, different architecture: independent binary classifiers (LightGBM + CatBoost) instead of joint matrix factorization.

## Results (private leaderboard)

| Model | Ranking | Private | Public |
|-------|---------|---------|--------|
| **CatBoost per-product** | **raw** | **0.02262** | **0.02230** |
| CatBoost per-product | percentile | 0.01877 | 0.01848 |
| LightGBM per-product | percentile | 0.01672 | 0.01642 |
| LightGBM per-product | raw | 0.01568 | 0.01548 |
| LightFM hybrid (companion) | — | **0.02502** | 0.02451 |
| Popularity baseline | — | 0.01813 | — |

Best result from this notebook beats popularity by +0.0045 but remains ~0.0024 below the LightFM hybrid.

## Key findings

1. **Raw probability beats percentile**  
   CatBoost raw vs percentile: +0.0039 on the private leaderboard.  
   Under `rank(pct=True)` the popularity fallback jumps from ~0.0005 to ~0.5 and leaks into the top-7.

2. **CatBoost is more stable than LightGBM on this dataset**  
   Same in-sample validation score (~0.0236), large external gap (−0.001 vs −0.008).  
   CatBoost’s regularisation + early stopping provides the stability that LightGBM defaults (and even a callback-only variant) do not.

3. **Per-product loses to joint factorization**  
   The gap is structural: 91 % of the target concentrates on a few co-occurring products. LightFM captures that co-occurrence; independent classifiers trained on a single-month snapshot cannot.

## Requirements

```bash
pip install catboost polars lightgbm pandas numpy scikit-learn
```

Custom transformers (`polars_transformers`) are assumed to be available in the environment (same set used by the LightFM repository).

## Structure

1. Schema diagnostics & curation  
2. Feature engineering (shared by both model families)  
3. Training: 8 LightGBM + 8 CatBoost classifiers  
4. Validation April → May  
5. Ranking experiment: raw vs percentile  
6. May → June submission  
7. Conclusions

## Companion repository

The stronger result (0.02500) was obtained with a LightFM hybrid model:

→ [Santander Product Recommendation — LightFM Hybrid](https://github.com/YOUR_USER/santander-lightfm)

## Path forward

- Full LightGBM regularisation stack (`num_leaves`, `min_child_samples`, `reg_lambda`) + early stopping  
- Rank-average blend of CatBoost raw + LightFM (likely 0.0255–0.026)  
- Skip percentile ranking in future experiments
