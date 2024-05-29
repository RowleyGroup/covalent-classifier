import argparse
import pandas as pd
import tensorflow as tf
from helpers import encode
from rdkit import Chem
from rdkit.DataStructs import TanimotoSimilarity, CosineSimilarity


REF_DF = pd.concat([
    pd.read_csv("./data/SMILES_training/trainingset_covalent_smiles.csv"),
    pd.read_csv("./data/SMILES_training/trainingset_noncovalent_smiles.csv")
])

def make_prediction(input_structure, model="./saved_models/GCNII",
                    get_similarity=False, ref_df=REF_DF):
    mol = Chem.MolFromSmiles(input_structure)

    if not mol:
        raise ValueError("Failed to generate mol from smiles")

    model = tf.keras.models.load_model(model)
    graph = encode(input_structure)
    if not get_similarity:
        return model.predict(graph)
    else:
        print("Calculating similarity...This might take some time...")
        sim = get_avg_similarity(input_structure, ref_df)
        return model.predict(graph), sim

def pairwise_distance(input_str, other_str):
    mol1 = Chem.MolFromSmiles(input_str)
    mol2 = Chem.MolFromSmiles(other_str)
    if mol1 is None or mol2 is None:
        return None
    return 1 - CosineSimilarity(Chem.RDKFingerprint(mol1), Chem.RDKFingerprint(mol2))

def tanimoto_similarity(input_str, other_str):
    mol1 = Chem.MolFromSmiles(input_str)
    mol2 = Chem.MolFromSmiles(other_str)
    if mol1 is None or mol2 is None:
        return None
    return TanimotoSimilarity(Chem.RDKFingerprint(mol1), Chem.RDKFingerprint(mol2))

def combined_metric(tanimoto_coefficient, pairwise_distance):
    return 0.5 * (tanimoto_coefficient + (1 / (1 + pairwise_distance)))

def get_avg_similarity(input_str, ref_df):
    similarity = 0
    pairwise = 0
    n = len(ref_df)
    for i in ref_df.SMILES.values:
        s = tanimoto_similarity(input_str, i)
        p = pairwise_distance(input_str, i)
        similarity += s
        pairwise += p
    return similarity / n, pairwise / n

def main():
    argParser = argparse.ArgumentParser()
    argParser.add_argument("inchi_string", help="SMILES string")
    argParser.add_argument("get_sim", help="Similarity metric", type=bool, default=False)
    args = argParser.parse_args()
    print(make_prediction(args.inchi_string, get_similarity=args.get_sim))


if __name__ == "__main__":
    main()