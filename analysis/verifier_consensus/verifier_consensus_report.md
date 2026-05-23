# Verifier Consensus Report

This is a non-network consensus over existing verifier features. It does not use CLIP/SigLIP yet.

## Tier Counts

| tier | count |
| --- | --- |
| A_lb_inferred | 2 |
| C_unresolved_one_of_three | 3 |
| E_low_priority | 123 |

## Top Consensus Rows

| id | consensus_rank | consensus_tier | consensus_fp_score | weak_label | soup_prob_pos | fp_risk_score | verifier_fp_score | top5_pos_frac | train_top10_pos_frac | alt_model_pos_frac | dinomlp_full_label | mytest_strict_label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 000088.jpg | 1 | A_lb_inferred | 0.4602 | inferred_fp | 0.7842 | 0.8619 | 0.8698 | 0.0000 | 0.1250 | 0.0000 | 0 | 0 |
| 000177.jpg | 2 | A_lb_inferred | 0.4370 | inferred_fp | 0.8164 | 0.8867 | 0.8804 | 0.0000 | 0.0000 | 0.1667 | 0 | 1 |
| 000131.jpg | 3 | C_unresolved_one_of_three | 0.4682 | one_of_three_fp | 0.6216 | 0.9257 | 0.9330 | 0.0000 | 0.0000 | 0.1667 | 0 | 0 |
| 000108.jpg | 4 | C_unresolved_one_of_three | 0.4305 | one_of_three_fp | 0.7046 | 0.9091 | 0.8939 | 0.0000 | 0.0000 | 0.3333 | 0 | 0 |
| 000124.jpg | 5 | C_unresolved_one_of_three | 0.4173 | one_of_three_fp | 0.8184 | 0.8863 | 0.8632 | 0.0000 | 0.0000 | 0.3333 | 0 | 1 |
| 000009.jpg | 6 | E_low_priority | 0.4998 |  | 0.9814 | 0.0537 | 0.0225 | 1.0000 | 1.0000 | 1.0000 | 1 | 1 |
| 000139.jpg | 7 | E_low_priority | 0.4974 |  | 0.5381 | 0.1424 | 0.2089 | 1.0000 | 1.0000 | 0.3333 | 1 | 0 |
| 000057.jpg | 8 | E_low_priority | 0.4971 |  | 0.9756 | 0.0549 | 0.0241 | 1.0000 | 1.0000 | 1.0000 | 1 | 1 |
| 000014.jpg | 9 | E_low_priority | 0.4934 |  | 0.9746 | 0.0551 | 0.0244 | 1.0000 | 1.0000 | 1.0000 | 1 | 1 |
| 000142.jpg | 10 | E_low_priority | 0.4926 |  | 0.8237 | 0.0853 | 0.1318 | 1.0000 | 1.0000 | 0.3333 | 1 | 0 |
| 000038.jpg | 11 | E_low_priority | 0.4917 |  | 0.9219 | 0.0656 | 0.1053 | 1.0000 | 1.0000 | 0.3333 | 1 | 0 |
| 000110.jpg | 12 | E_low_priority | 0.4896 |  | 0.9736 | 0.0553 | 0.0246 | 1.0000 | 1.0000 | 1.0000 | 1 | 1 |
| 000055.jpg | 13 | E_low_priority | 0.4861 |  | 0.9717 | 0.0557 | 0.0251 | 1.0000 | 1.0000 | 1.0000 | 1 | 1 |
| 000102.jpg | 14 | E_low_priority | 0.4825 |  | 0.9702 | 0.0560 | 0.0255 | 1.0000 | 1.0000 | 1.0000 | 1 | 1 |
| 000019.jpg | 15 | E_low_priority | 0.4768 |  | 0.9697 | 0.0561 | 0.0257 | 1.0000 | 1.0000 | 1.0000 | 1 | 1 |
| 000157.jpg | 16 | E_low_priority | 0.4768 |  | 0.9697 | 0.0561 | 0.0257 | 1.0000 | 1.0000 | 1.0000 | 1 | 1 |
| 000033.jpg | 17 | E_low_priority | 0.4710 |  | 0.9692 | 0.0562 | 0.0258 | 1.0000 | 1.0000 | 1.0000 | 1 | 1 |
| 000116.jpg | 18 | E_low_priority | 0.4677 |  | 0.9458 | 0.0608 | 0.0655 | 1.0000 | 1.0000 | 0.6667 | 1 | 1 |
| 000051.jpg | 19 | E_low_priority | 0.4652 |  | 0.9688 | 0.0563 | 0.0259 | 1.0000 | 1.0000 | 1.0000 | 1 | 1 |
| 000060.jpg | 20 | E_low_priority | 0.4652 |  | 0.9688 | 0.0563 | 0.0259 | 1.0000 | 1.0000 | 1.0000 | 1 | 1 |
| 000194.jpg | 21 | E_low_priority | 0.4645 |  | 0.5615 | 0.1655 | 0.2178 | 1.0000 | 0.8889 | 0.5000 | 1 | 0 |
| 000046.jpg | 22 | E_low_priority | 0.4604 |  | 0.6582 | 0.6784 | 0.7491 | 0.2000 | 0.2000 | 0.1667 | 1 | 0 |
| 000175.jpg | 23 | E_low_priority | 0.4597 |  | 0.9673 | 0.0565 | 0.0263 | 1.0000 | 1.0000 | 1.0000 | 1 | 1 |
| 000164.jpg | 24 | E_low_priority | 0.4578 |  | 0.8096 | 0.1631 | 0.1618 | 1.0000 | 1.0000 | 0.3333 | 1 | 0 |
| 000093.jpg | 25 | E_low_priority | 0.4565 |  | 0.9634 | 0.0573 | 0.0274 | 1.0000 | 1.0000 | 1.0000 | 1 | 1 |
| 000020.jpg | 26 | E_low_priority | 0.4562 |  | 0.8853 | 0.8329 | 0.8445 | 0.0000 | 0.1000 | 0.0000 | 0 | 0 |
| 000103.jpg | 27 | E_low_priority | 0.4540 |  | 0.9565 | 0.0587 | 0.0292 | 1.0000 | 1.0000 | 1.0000 | 1 | 1 |
| 000134.jpg | 28 | E_low_priority | 0.4508 |  | 0.9531 | 0.0594 | 0.0302 | 1.0000 | 1.0000 | 1.0000 | 1 | 1 |
| 000105.jpg | 29 | E_low_priority | 0.4461 |  | 0.5703 | 0.1359 | 0.1502 | 1.0000 | 1.0000 | 0.8333 | 1 | 0 |
| 000001.jpg | 30 | E_low_priority | 0.4453 |  | 0.9512 | 0.0598 | 0.0307 | 1.0000 | 1.0000 | 1.0000 | 1 | 1 |
| 000166.jpg | 31 | E_low_priority | 0.4453 |  | 0.9512 | 0.0598 | 0.0307 | 1.0000 | 1.0000 | 1.0000 | 1 | 1 |
| 000169.jpg | 32 | E_low_priority | 0.4446 |  | 0.9043 | 0.0691 | 0.0767 | 1.0000 | 1.0000 | 0.6667 | 1 | 1 |
| 000070.jpg | 33 | E_low_priority | 0.4423 |  | 0.5439 | 0.4726 | 0.4638 | 0.8000 | 0.7143 | 0.0000 | 0 | 0 |
| 000062.jpg | 34 | E_low_priority | 0.4399 |  | 0.8247 | 0.6473 | 0.7227 | 0.2000 | 0.1111 | 0.1667 | 0 | 0 |
| 000058.jpg | 35 | E_low_priority | 0.4396 |  | 0.9502 | 0.0600 | 0.0309 | 1.0000 | 1.0000 | 1.0000 | 1 | 1 |

## Candidate Submissions

- A only: analysis/verifier_consensus/candidate_submissions/submission_consensus_A.csv
- A+B: analysis/verifier_consensus/candidate_submissions/submission_consensus_A_B.csv

## Interpretation

- Tier A is backed by leaderboard arithmetic and remains the only clean next submission candidate.
- Tier B excludes the unresolved 108/124/131 group, but is still untested and should wait for manual review or CLIP/SigLIP evidence.
- Tier C is deliberately separated because exactly one of 108,124,131 appears false-positive, but the identity is unresolved.
