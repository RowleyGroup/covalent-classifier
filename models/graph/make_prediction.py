import argparse
import tensorflow as tf
from .helpers import encode


def make_prediction(input_structure, model="./saved_models/GCNII"):
    model = tf.keras.models.load_model(model)
    graph = encode(input_structure)
    return model.predict(graph)


def main():
    argParser = argparse.ArgumentParser()
    argParser.add_argument("inchi_string", help="InChI or SMILES string")
    args = argParser.parse_args()
    print(make_prediction(args.inchi_string), flush=True)


if __name__ == "__main__":
    main()