import pandas as pd 

df_train = pd.read_csv("./training_data_all.csv")
df_test = pd.read_csv("./test_data_all.csv")

df_train = df_train.drop_duplicates(subset=["InChI"])
df_test = df_test.drop_duplicates(subset=["InChI"])

assert not (set(df_train.query("covalent == 1").InChI).intersection(df_train.query("covalent == 0").InChI))
assert not (set(df_test.query("covalent == 1").InChI).intersection(df_test.query("covalent == 0").InChI))
overlap = set(df_train.InChI).intersection(df_test.InChI)
df_test = df_test[~df_test.InChI.isin(overlap)]

# check again
assert not (set(df_train.query("covalent == 1").InChI).intersection(df_train.query("covalent == 0").InChI))
assert not (set(df_test.query("covalent == 1").InChI).intersection(df_test.query("covalent == 0").InChI))
assert not (set(df_train.InChI).intersection(df_test.InChI))

# df_train.to_csv("./training_data_all.csv", index=False)
# df_test.to_csv("./test_data_all.csv", index=False)
