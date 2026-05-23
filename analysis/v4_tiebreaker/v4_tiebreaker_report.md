# Testlike V4 Tie-breaker Report

V4 gate: cluster >= 0.993, top >= 1.0
Runs evaluated: 68
Runs passing gate: 29
Runs saturated at 1.0/1.0: 15

## Known Kaggle Outcomes Inside V4 Gate

| run_tag | v4_mean_f1 | known_test_f1 | submission_pos | diff_vs_baseline | baseline_pos_to_neg | baseline_neg_to_pos | uses_mytest_supervision | hidden_risk_flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| myval_v13_hi288_seed42_soup/soup.pt | 0.9969 | 0.7196 | 128.0000 | 0.0000 | 0.0000 | 0.0000 | False |  |
| mytest_augment_v2/soup_top3.pt | 1.0000 | 0.6702 | 108.0000 | 30.0000 | 25.0000 | 5.0000 | True | mytest_supervised;pos_count_far; |

## Comparable Submissions Passing V4 Gate

| run_tag | v4_mean_f1 | known_test_f1 | submission_pos | diff_vs_baseline | baseline_pos_to_neg | baseline_neg_to_pos | uses_mytest_supervision | hidden_risk_flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| myval_v13_hi288_seed42_soup/soup.pt | 0.9969 | 0.7196 | 128.0000 | 0.0000 | 0.0000 | 0.0000 | False |  |
| mytest_strict_dino_v1/best.pt | 1.0000 |  | 104.0000 | 28.0000 | 26.0000 | 2.0000 | True | mytest_supervised;pos_count_far; |
| mytest_augment_v1/best.pt | 1.0000 |  | 111.0000 | 29.0000 | 23.0000 | 6.0000 | True | mytest_supervised; |
| mytest_augment_v2/best.pt | 1.0000 |  | 108.0000 | 30.0000 | 25.0000 | 5.0000 | True | mytest_supervised;pos_count_far; |
| mytest_augment_v2/soup_top3.pt | 1.0000 | 0.6702 | 108.0000 | 30.0000 | 25.0000 | 5.0000 | True | mytest_supervised;pos_count_far; |

## Interpretation

- V4 is a useful gate but not a sufficient ranking metric: known Kaggle regressions pass or saturate it.
- Prefer V4-passing candidates with small submission diffs from current best unless a diff is backed by leaderboard arithmetic.
- Penalize mytest-supervised runs even when V4-perfect; historical Kaggle evidence shows domain-shift regressions.
- The next submission-side candidate remains current best plus inferred force-zero 88,177 only.
