#%%
import exmol
from models.graph.helpers import encode, encoder
import tensorflow as tf
# model = tf.keras.models.load_model("./saved_models/GCNII")
# base = 'Cc1onc(-c2ccccc2Cl)c1C(=O)NC1C(=O)N2C1SC(C)(C)C2C(=O)O'

# futibatinib
base = "COC1=CC(=CC(OC)=C1)C#CC1=NN([C@H]2CCN(C2)C(=O)C=C)C2=C1C(N)=NC=N2"

class Predict:
    def __init__(self):
        self.model = tf.keras.models.load_model("./saved_models/GraphSage")
    def __call__(self, smiles_list):
        graphs = encoder(smiles_list)
        return self.model.predict(graphs)
p = Predict()

samples = exmol.sample_space(base, p, batched=True, preset="narrow")
# %%
cfs = exmol.cf_explain(samples, nmols=5)
exmol.plot_cf(cfs, mol_size=(300, 300), mol_fontsize=10)

# %%
cfs = exmol.cf_explain(samples)
exmol.plot_space(samples, cfs)
# %%
