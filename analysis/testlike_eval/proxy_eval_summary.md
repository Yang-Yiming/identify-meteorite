# Test-Like Proxy Evaluation

Historical checkpoints evaluated on myval and first-pass test-like validation sets.

## Summary

| run | known_test_f1 | known_myval_f1_from_docs | f1_at_0_5__myval_masked | f1_at_0_5__testlike_cluster_v1 | f1_at_0_5__testlike_top_v1 | best_f1__testlike_cluster_v1 | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| soup_reduced_notstone | 0.7196 | 0.7251 | 0.7230 | 0.9640 | 0.9499 | 0.9706 | current SOTA: soup + reduced not-stone post-process |
| soup_old_notstone | 0.6986 | 0.7251 | 0.7230 | 0.9640 | 0.9499 | 0.9706 | same checkpoint, old aggressive not-stone post-process |
| mytest_augment_soup | 0.6702 | 0.7688 | 0.7688 | 0.9757 | 0.9684 | 0.9853 | mytest merged into train, top-3 soup |
| mytest_split_protocol | 0.6598 | 0.7321 | 0.7278 | 0.9642 | 0.9547 | 0.9853 | mytest train+val protocol |
| splitval_augment_soup | 0.6321 | 0.7446 | 0.7423 | 0.9637 | 0.9567 | 0.9751 | mytest merged, internal split validation, top-3 soup |
| mytest_pretrain_finetune | 0.5521 | 0.7358 | 0.7333 | 0.9104 | 0.9448 | 0.9163 | supervised mytest pretrain then original-data finetune |


## Rank Correlation With Known Test F1

| dataset | metric | spearman_like_rank_corr |
| --- | --- | --- |
| myval_masked | f1_at_0_5 | -0.5798 |
| myval_masked | best_f1 | -0.4638 |
| myval_masked | prob_mean | 0.8117 |
| testlike_cluster_v1 | f1_at_0_5 | 0.5218 |
| testlike_cluster_v1 | best_f1 | 0.0883 |
| testlike_cluster_v1 | prob_mean | 0.6377 |
| testlike_top_v1 | f1_at_0_5 | 0.0580 |
| testlike_top_v1 | best_f1 | 0.0580 |
| testlike_top_v1 | prob_mean | -0.2319 |


Note: `soup_reduced_notstone` and `soup_old_notstone` share the same checkpoint, so checkpoint-only proxy metrics cannot distinguish their different post-process policies.

## Interpretation

V1 is **not yet a reliable model-selection proxy**. It is directionally better
than myval on `f1_at_0_5` rank correlation for this small run set, but it still
assigns very high F1 to mytest-heavy models that are known to fail on Kaggle
test. In particular, `mytest_augment_soup` scores highest on both
`myval_masked` and `testlike_cluster_v1`, while its known test F1 is far below
the current soup + reduced-not-stone result.

The likely failure mode is that V1 lightweight image statistics select samples
that are easy for all ConvNeXt-family checkpoints and still do not encode the
important train/test domain mismatch. The next iteration should add semantic
frozen features, at minimum the current ConvNeXt penultimate embedding, then
DINO/CLIP/SigLIP if available.

Use these files as report data, but do not use V1 alone to choose submissions.
