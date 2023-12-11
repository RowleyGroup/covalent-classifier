#%%
import pandas as pd
from rdkit import Chem

df1 = pd.read_csv("./data/SMILES_test/testset_positive.csv")
df2 = pd.read_csv("./data/SMILES_test/testset_negative.csv")
df3 = pd.concat([df1, df2])
df3 = df3[["SMILES", "InChI", "warhead_category", "covalent"]]

df4 = pd.read_csv("./data/SMILES_test/testset_decoy.csv")
df4["InChI"] = df4.SMILES.apply(lambda x: Chem.MolToInchi(Chem.MolFromSmiles(x)))
df4["warhead_category"] = "noncovalentdecoy"
df4["covalent"] = 0
df4 = df4.drop([df4.columns[0]], axis=1)
df = pd.concat([df3, df4])
df = df.drop_duplicates(subset=["InChI"])
df.to_csv("test_data_all.csv", index=False)
# %%
