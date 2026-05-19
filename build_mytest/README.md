meteorite: from `https://encyclopedia-of-meteorites.com/`
rock: from two kaggle datasets, `https://www.kaggle.com/api/v1/datasets/download/neelgajare/rocks-dataset` and `https://www.kaggle.com/api/v1/datasets/download/salmaneunus/rock-classification`

filter:
1. 用 SAM3 找 mask，找不到或者 mask 所占面积比例 < 0.005 的都 filter 掉。
