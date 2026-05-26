# Test-Like Proxy Evaluation

Historical checkpoints evaluated on myval and first-pass test-like validation sets.

## Summary

| run | known_test_f1 | known_myval_f1_from_docs | f1_at_0_5__myval_masked | f1_at_0_5__testlike_cluster_dino_v4 | f1_at_0_5__testlike_top_dino_v4 | best_f1__testlike_cluster_dino_v4 | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| soup_reduced_notstone | 0.7196 | 0.7251 | 0.7230 | 0.9937 | 1.0000 | 0.9937 | current SOTA: soup + reduced not-stone post-process |
| soup_old_notstone | 0.6986 | 0.7251 | 0.7230 | 0.9937 | 1.0000 | 0.9937 | same checkpoint, old aggressive not-stone post-process |
| mytest_augment_soup | 0.6702 | 0.7688 | 0.7688 | 1.0000 | 1.0000 | 1.0000 | mytest merged into train, top-3 soup |
| mytest_split_protocol | 0.6598 | 0.7321 | 0.7278 | 0.9814 | 0.9877 | 0.9937 | mytest train+val protocol |
| splitval_augment_soup | 0.6321 | 0.7446 | 0.7423 | 0.9814 | 0.9816 | 0.9936 | mytest merged, internal split validation, top-3 soup |
| mytest_pretrain_finetune | 0.5521 | 0.7358 | 0.7333 | 0.8725 | 0.9487 | 0.8974 | supervised mytest pretrain then original-data finetune |
| mytest_strict_dino_v1 |  | 0.7202 | 0.7163 | 1.0000 | 1.0000 | 1.0000 | original train + strict DINO-filtered mytest, sample_weight=0.5 |
| cnv2_tiny |  | 0.6736 | 0.6701 | 0.9383 | 0.9524 | 0.9487 | ConvNeXt V2 tiny, worse than soup |


## Rank Correlation With Known Test F1

| dataset | metric | spearman_like_rank_corr |
| --- | --- | --- |
| myval_masked | f1_at_0_5 | -0.5798 |
| myval_masked | best_f1 | -0.4638 |
| myval_masked | prob_mean | 0.8117 |
| testlike_cluster_dino_v4 | f1_at_0_5 | 0.7945 |
| testlike_cluster_dino_v4 | best_f1 | 0.6983 |
| testlike_cluster_dino_v4 | prob_mean | -0.0580 |
| testlike_top_dino_v4 | f1_at_0_5 | 0.9411 |
| testlike_top_dino_v4 | best_f1 | 0.8452 |
| testlike_top_dino_v4 | prob_mean | -0.0580 |


Note: `soup_reduced_notstone` and `soup_old_notstone` share the same checkpoint, so checkpoint-only proxy metrics cannot distinguish their different post-process policies.