import argparse
import tensorflow as tf
from molgraph.models import GradientActivationMapping
from molgraph.chemistry import vis
from config import encode


def make_gradcam_heatmap(input_structure, model="./saved_models/GCNII"):
    model = tf.keras.models.load_model(model)
    gam_model = GradientActivationMapping(
        model,
        [i.name for i in model.layers if "conv" in i.name], # all conv layers by default
        output_activation=None,
        discard_negative_values=True
    )
    graph = encode(input_structure)
    gam = gam_model(graph)
    heatmap = vis.visualize_maps(molecule=input_structure, maps=gam[0])
    heatmap.save("./gradcam_heatmap.png")


def main():
    argParser = argparse.ArgumentParser()
    argParser.add_argument("inchi_string", help="InChI or SMILES string")
    args = argParser.parse_args()
    make_gradcam_heatmap(args.inchi_string)


if __name__ == "__main__":
    main()