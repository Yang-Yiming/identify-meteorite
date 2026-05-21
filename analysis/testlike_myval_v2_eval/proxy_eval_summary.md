# Test-Like Proxy Evaluation

Historical checkpoints evaluated on myval and first-pass test-like validation sets.

## Summary

| run | known_test_f1 | known_myval_f1_from_docs | f1_at_0_5__myval_masked | f1_at_0_5__testlike_cluster_myval_v2 | f1_at_0_5__testlike_top_myval_v2 | best_f1__testlike_cluster_myval_v2 | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| soup_reduced_notstone | 0.7196 | 0.7251 | 0.7230 | 0.7607 | 0.8045 | 0.7654 | current SOTA: soup + reduced not-stone post-process |
| soup_old_notstone | 0.6986 | 0.7251 | 0.7230 | 0.7607 | 0.8045 | 0.7654 | same checkpoint, old aggressive not-stone post-process |
| mytest_augment_soup | 0.6702 | 0.7688 | 0.7688 | 0.8243 | 0.8214 | 0.8243 | mytest merged into train, top-3 soup |
| mytest_split_protocol | 0.6598 | 0.7321 | 0.7278 | 0.7516 | 0.7841 | 0.7852 | mytest train+val protocol |
| splitval_augment_soup | 0.6321 | 0.7446 | 0.7423 | 0.7632 | 0.7811 | 0.7972 | mytest merged, internal split validation, top-3 soup |
| mytest_pretrain_finetune | 0.5521 | 0.7358 | 0.7333 | 0.7518 | 0.7607 | 0.7591 | supervised mytest pretrain then original-data finetune |


## Rank Correlation With Known Test F1

| dataset | metric | spearman_like_rank_corr |
| --- | --- | --- |
| myval_masked | f1_at_0_5 | -0.5798 |
| myval_masked | best_f1 | -0.4638 |
| myval_masked | prob_mean | 0.8117 |
| testlike_cluster_myval_v2 | f1_at_0_5 | 0.2319 |
| testlike_cluster_myval_v2 | best_f1 | 0.0580 |
| testlike_cluster_myval_v2 | prob_mean | 0.8117 |
| testlike_top_myval_v2 | f1_at_0_5 | 0.8117 |
| testlike_top_myval_v2 | best_f1 | 0.8117 |
| testlike_top_myval_v2 | prob_mean | 0.9276 |


Note: `soup_reduced_notstone` and `soup_old_notstone` share the same checkpoint, so checkpoint-only proxy metrics cannot distinguish their different post-process policies.

## Interpretation

Myval-only V2 removes direct train/mytest contamination from the proxy
construction. Absolute F1 values are now in a plausible range, unlike V1's
near-saturated 0.96+ scores.

However, V2 is still not a reliable final model-selection metric. The
mytest-augmented soup remains highest on both myval and the myval-only
test-like subsets, despite its known Kaggle test F1 being substantially below
the current soup + reduced-not-stone result. This suggests that myval itself is
too small, too noisy, or too close to the clean mytest-style domain to fully
represent Kaggle test.

The useful takeaway is negative but actionable: myval-only test-likeness is a
cleaner diagnostic than V1, but the next useful version likely needs stronger
semantic embeddings and/or carefully audited mytest/test-neighbor evidence
rather than only lightweight image statistics over myval.
