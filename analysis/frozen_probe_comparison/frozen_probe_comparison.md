# Frozen Feature Probe Comparison

Simple frozen-feature logistic probes on Testlike V4.

## Top Rows

run,C,class_weight,v4_cluster_f1_at_0_5,v4_top_f1_at_0_5,myval_f1_at_0_5,test_pos_pred_at_0_5,diff_vs_baseline,baseline_pos_to_neg,baseline_neg_to_pos
clip_balanced,3.0,balanced,1.0,1.0,0.6139817629179332,110,48,33,15
siglip_balanced,10.0,balanced,1.0,1.0,0.6328358208955224,103,49,37,12
siglip_noweight,10.0,none,1.0,1.0,0.6328358208955224,103,49,37,12
clip_balanced,10.0,balanced,1.0,1.0,0.6116207951070336,110,54,36,18
siglip_balanced,3.0,balanced,0.9937106918238994,0.9937888198757764,0.6369047619047619,103,49,37,12
siglip_noweight,3.0,none,0.9937106918238994,0.9937888198757764,0.6328358208955224,103,49,37,12
siglip_noweight,1.0,none,0.9813664596273292,0.9937888198757764,0.6548672566371682,105,45,34,11
siglip_balanced,1.0,balanced,0.9813664596273292,0.9937888198757764,0.6548672566371682,103,47,36,11
clip_balanced,1.0,balanced,0.981132075471698,0.9876543209876544,0.6073619631901841,110,48,33,15
siglip_balanced,0.3,balanced,0.9565217391304348,0.98159509202454,0.6468842729970327,103,49,37,12
siglip_noweight,0.3,none,0.951219512195122,0.98159509202454,0.6449704142011834,105,47,35,12
siglip_noweight,0.1,none,0.9454545454545454,0.98159509202454,0.6390532544378699,104,46,35,11


## Interpretation

- SigLIP and CLIP can both saturate V4 with a simple logistic head, so model/representation capacity is a real direction.
- The V4-perfect probes are much more conservative than current best: 103-110 positives vs 128 and 48-54 label diffs.
- This is not submission-ready, but it is the right kind of simple experiment: strong frozen backbone + shallow head, selected by V4 and sanity-checked by behavior.
- Next model-capability steps should try calibration/thresholding toward current positive count, shallow MLP probes, or fine-tuning/adapters, rather than manual FP list expansion.
