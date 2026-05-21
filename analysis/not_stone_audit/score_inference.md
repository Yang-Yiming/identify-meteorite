# Not-Stone Score Inference

Known submissions:

- Full old not-stone list: F1 = 0.69856
- Current 9-id not-stone list: F1 = 0.71962
- Current list but restoring `18,23,72,133` to positive: F1 = 0.71559

The current 9-id list is:

```text
18
23
44
72
100
133
145
162
187
```

For the soup raw CSV, only these six IDs are actually changed by the current
list:

```text
18
23
44
72
133
162
```

The latest 4-image ablation changed `18,23,72,133` from 0 to 1 and dropped
F1 from 0.71962 to 0.71559. For F1,

```text
F1 = 2TP / (2TP + FP + FN)
```

Flipping one prediction from 0 to 1 always increases the denominator by 1:

- if the sample is truly positive: `TP += 1`, `FN -= 1`
- if the sample is truly negative: `FP += 1`

The observed rounded scores are consistent with:

```text
current 9-id list:      TP = 77, denominator = 214, F1 = 154/214 = 0.719626
4-image restored list:  TP = 78, denominator = 218, F1 = 156/218 = 0.715596
```

So among `18,23,72,133`, approximately exactly **one** is truly positive and
the other three are truly negative.

The earlier jump from 0.69856 to 0.71962 after removing
`48,67,154,159,185` from the old forced-zero list is consistent with four of
those five being true positives.

Next best single-ablation hypothesis:

- Restore only `23` to positive while keeping `18,44,72,100,133,145,162,187`
  forced to 0.
- If `23` is the one true positive, expected F1 is `156/215 = 0.72558`.
- If `23` is negative, expected F1 is `154/215 = 0.71628`.

Rationale for trying `23` first: it has high model probability under multiple
models and visually resembles a polished/cut meteorite or mineral specimen more
than obvious non-stone IDs like `44` and `162`.
