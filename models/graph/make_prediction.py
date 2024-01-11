import argparse
import tensorflow as tf
from helpers import encode
from rdkit import Chem


def make_prediction(input_structure, model="./saved_models/GCNII"):
    mol = Chem.MolFromSmiles(input_structure)
    if not mol:
        raise ValueError("Failed to generate mol from smiles")
    model = tf.keras.models.load_model(model)
    graph = encode(input_structure)
    return model.predict(graph)


def main():
    argParser = argparse.ArgumentParser()
    argParser.add_argument("inchi_string", help="SMILES string")
    args = argParser.parse_args()
    print(make_prediction(args.inchi_string), flush=True)


if __name__ == "__main__":
    main()