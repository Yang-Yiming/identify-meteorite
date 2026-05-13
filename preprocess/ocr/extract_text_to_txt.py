import pandas as pd

input_csv = "preprocess/test_text.csv"
output_txt = "preprocess/text.csv"

df = pd.read_csv(input_csv)

one_list = ["陨石", "meteorite", "meteor"]
zero_list = ["岩石"]

with open(output_txt, "w") as f:
    for img, cap in zip(df["img"], df["cap"]):
        for one_word in one_list:
            if one_word in cap:
                f.write(f"{img},1\n")
        for zero_word in zero_list:
            if zero_word in cap:
                f.write(f"{img},0\n")
