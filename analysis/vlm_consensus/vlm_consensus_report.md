# VLM Consensus Report

Combines DINO verifier features with SigLIP and CLIP nearest-neighbor label ratios.

## Top 40 Candidates

| id | weak_label | consensus_note | soup_prob_pos | top5_pos_frac | siglip_top5_pos_frac | clip_top5_pos_frac | three_embedding_neg_consensus | alt_model_pos_frac | verifier_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 000131.jpg | one_of_three_fp | unresolved_group_mixed_vlm | 0.6216 | 0.0000 | 0.0000 | 0.4000 | 0.8667 | 0.1667 | 1 |
| 000108.jpg | one_of_three_fp | unresolved_group_mixed_vlm | 0.7046 | 0.0000 | 0.0000 | 0.4000 | 0.8667 | 0.3333 | 2 |
| 000177.jpg | inferred_fp | lb_inferred_fp_and_embedding_negative | 0.8164 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.1667 | 3 |
| 000088.jpg | inferred_fp | lb_inferred_fp_and_embedding_negative | 0.7842 | 0.0000 | 0.2000 | 0.2000 | 0.8667 | 0.0000 | 4 |
| 000124.jpg | one_of_three_fp | unresolved_group_clip_positive | 0.8184 | 0.0000 | 0.4000 | 0.8000 | 0.6000 | 0.3333 | 5 |
| 000020.jpg |  |  | 0.8853 | 0.0000 | 0.2000 | 0.8000 | 0.6667 | 0.0000 | 6 |
| 000106.jpg |  |  | 0.8481 | 0.0000 | 0.0000 | 0.6000 | 0.8000 | 0.1667 | 7 |
| 000082.jpg |  |  | 0.9458 | 0.0000 | 0.4000 | 0.6000 | 0.6667 | 0.6667 | 8 |
| 000138.jpg |  |  | 0.8315 | 0.0000 | 0.4000 | 0.8000 | 0.6000 | 0.6667 | 9 |
| 000035.jpg |  |  | 0.9429 | 0.0000 | 0.6000 | 0.4000 | 0.6667 | 0.8333 | 10 |
| 000053.jpg |  |  | 0.8672 | 0.0000 | 0.4000 | 0.2000 | 0.8000 | 0.8333 | 11 |
| 000046.jpg |  |  | 0.6582 | 0.2000 | 0.6000 | 0.2000 | 0.6667 | 0.1667 | 12 |
| 000062.jpg |  |  | 0.8247 | 0.2000 | 0.2000 | 0.4000 | 0.7333 | 0.1667 | 13 |
| 000016.jpg |  |  | 0.8745 | 0.2000 | 0.0000 | 0.6000 | 0.7333 | 0.8333 | 14 |
| 000160.jpg |  |  | 0.7749 | 0.4000 | 0.2000 | 0.6000 | 0.6000 | 0.6667 | 15 |
| 000127.jpg |  |  | 0.7783 | 0.4000 | 0.2000 | 0.2000 | 0.7333 | 0.8333 | 16 |
| 000186.jpg |  |  | 0.7358 | 0.4000 | 0.6000 | 0.8000 | 0.4000 | 0.6667 | 17 |
| 000117.jpg |  |  | 0.9312 | 0.2000 | 0.6000 | 1.0000 | 0.4000 | 1.0000 | 18 |
| 000066.jpg |  |  | 0.9424 | 0.2000 | 0.0000 | 0.6000 | 0.7333 | 1.0000 | 19 |
| 000006.jpg |  |  | 0.6133 | 0.4000 | 0.6000 | 0.6000 | 0.4667 | 1.0000 | 20 |
| 000130.jpg |  |  | 0.9370 | 0.2000 | 0.8000 | 0.8000 | 0.4000 | 1.0000 | 21 |
| 000017.jpg |  |  | 0.9346 | 0.2000 | 0.4000 | 1.0000 | 0.4667 | 1.0000 | 22 |
| 000104.jpg |  |  | 0.9409 | 0.4000 | 0.6000 | 0.4000 | 0.5333 | 0.6667 | 23 |
| 000084.jpg |  |  | 0.6987 | 0.8000 | 0.6000 | 0.4000 | 0.4000 | 0.5000 | 24 |
| 000086.jpg |  |  | 0.9214 | 0.2000 | 1.0000 | 0.4000 | 0.4667 | 0.5000 | 25 |
| 000151.jpg |  |  | 0.7095 | 0.4000 | 1.0000 | 1.0000 | 0.2000 | 0.6667 | 26 |
| 000045.jpg |  |  | 0.9380 | 0.4000 | 0.2000 | 0.2000 | 0.7333 | 0.8333 | 27 |
| 000099.jpg |  |  | 0.8945 | 0.4000 | 0.6000 | 0.8000 | 0.4000 | 1.0000 | 28 |
| 000037.jpg |  |  | 0.5913 | 0.6000 | 0.8000 | 0.6000 | 0.3333 | 0.8333 | 29 |
| 000039.jpg |  |  | 0.9888 | 0.4000 | 0.6000 | 0.6000 | 0.4667 | 1.0000 | 30 |
| 000070.jpg |  |  | 0.5439 | 0.8000 | 0.6000 | 1.0000 | 0.2000 | 0.0000 | 31 |
| 000173.jpg |  |  | 0.7217 | 0.6000 | 0.8000 | 1.0000 | 0.2000 | 0.8333 | 32 |
| 000119.jpg |  |  | 0.7144 | 0.6000 | 0.6000 | 0.4000 | 0.4667 | 0.1667 | 33 |
| 000021.jpg |  |  | 0.8804 | 0.6000 | 0.4000 | 0.6000 | 0.4667 | 1.0000 | 34 |
| 000121.jpg |  |  | 0.9463 | 0.2000 | 0.8000 | 1.0000 | 0.3333 | 0.6667 | 35 |
| 000056.jpg |  |  | 0.6587 | 0.8000 | 0.4000 | 0.8000 | 0.3333 | 0.0000 | 36 |
| 000118.jpg |  |  | 0.9229 | 0.4000 | 0.8000 | 0.6000 | 0.4000 | 1.0000 | 37 |
| 000061.jpg |  |  | 0.9600 | 0.6000 | 0.8000 | 0.8000 | 0.2667 | 1.0000 | 38 |
| 000136.jpg |  |  | 0.9404 | 0.6000 | 0.0000 | 0.8000 | 0.5333 | 1.0000 | 39 |
| 000068.jpg |  |  | 0.7808 | 0.4000 | 0.6000 | 0.8000 | 0.4000 | 0.8333 | 40 |

## Three-embedding Strong Negatives

| id | weak_label | consensus_note | soup_prob_pos | top5_pos_frac | siglip_top5_pos_frac | clip_top5_pos_frac | three_embedding_neg_consensus | alt_model_pos_frac | verifier_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 000177.jpg | inferred_fp | lb_inferred_fp_and_embedding_negative | 0.8164 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.1667 | 3 |
| 000088.jpg | inferred_fp | lb_inferred_fp_and_embedding_negative | 0.7842 | 0.0000 | 0.2000 | 0.2000 | 0.8667 | 0.0000 | 4 |

## Unresolved 108/124/131 Group

| id | weak_label | consensus_note | soup_prob_pos | top5_pos_frac | siglip_top5_pos_frac | clip_top5_pos_frac | three_embedding_neg_consensus | alt_model_pos_frac | verifier_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 000131.jpg | one_of_three_fp | unresolved_group_mixed_vlm | 0.6216 | 0.0000 | 0.0000 | 0.4000 | 0.8667 | 0.1667 | 1 |
| 000108.jpg | one_of_three_fp | unresolved_group_mixed_vlm | 0.7046 | 0.0000 | 0.0000 | 0.4000 | 0.8667 | 0.3333 | 2 |
| 000124.jpg | one_of_three_fp | unresolved_group_clip_positive | 0.8184 | 0.0000 | 0.4000 | 0.8000 | 0.6000 | 0.3333 | 5 |

## Interpretation

- 88 and 177 remain the only clean next-submission FP candidates: leaderboard arithmetic and DINO/SigLIP/CLIP all support negative evidence.
- 124 becomes much less attractive to zero because CLIP top5 is strongly positive.
- 108 and 131 remain unresolved: DINO/SigLIP are negative, but CLIP is mixed, and leaderboard arithmetic says only one of 108/124/131 is FP.
- New candidates such as 20/106 should not be submitted before manual visual review or additional leaderboard evidence.
