# Test-Like Proxy Evaluation

Historical checkpoints evaluated on myval and first-pass test-like validation sets.

## Summary

| run | known_test_f1 | known_myval_f1_from_docs | f1_at_0_5__myval_masked | f1_at_0_5__testlike_cluster_dino_v3_with_strict | f1_at_0_5__testlike_top_dino_v3_with_strict | best_f1__testlike_cluster_dino_v3_with_strict | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| soup_reduced_notstone | 0.7196 | 0.7251 | 0.7230 | 0.7709 | 0.8045 | 0.7709 | current SOTA: soup + reduced not-stone post-process |
| soup_old_notstone | 0.6986 | 0.7251 | 0.7230 | 0.7709 | 0.8045 | 0.7709 | same checkpoint, old aggressive not-stone post-process |
| mytest_augment_soup | 0.6702 | 0.7688 | 0.7688 | 0.8066 | 0.8182 | 0.8156 | mytest merged into train, top-3 soup |
| mytest_split_protocol | 0.6598 | 0.7321 | 0.7278 | 0.7826 | 0.7978 | 0.8187 | mytest train+val protocol |
| splitval_augment_soup | 0.6321 | 0.7446 | 0.7423 | 0.7889 | 0.8182 | 0.8000 | mytest merged, internal split validation, top-3 soup |
| mytest_pretrain_finetune | 0.5521 | 0.7358 | 0.7333 | 0.7927 | 0.8199 | 0.7927 | supervised mytest pretrain then original-data finetune |
| mytest_strict_dino_v1 |  | 0.7202 | 0.7163 | 0.7619 | 0.7892 | 0.7630 | original train + strict DINO-filtered mytest, sample_weight=0.5 |


## Rank Correlation With Known Test F1

| dataset | metric | spearman_like_rank_corr |
| --- | --- | --- |
| myval_masked | f1_at_0_5 | -0.5798 |
| myval_masked | best_f1 | -0.4638 |
| myval_masked | prob_mean | 0.8117 |
| testlike_cluster_dino_v3_with_strict | f1_at_0_5 | -0.6377 |
| testlike_cluster_dino_v3_with_strict | best_f1 | -0.4638 |
| testlike_cluster_dino_v3_with_strict | prob_mean | 0.4638 |
| testlike_top_dino_v3_with_strict | f1_at_0_5 | -0.5885 |
| testlike_top_dino_v3_with_strict | best_f1 | -0.5798 |
| testlike_top_dino_v3_with_strict | prob_mean | 0.6377 |


Note: `soup_reduced_notstone` and `soup_old_notstone` share the same checkpoint, so checkpoint-only proxy metrics cannot distinguish their different post-process policies.