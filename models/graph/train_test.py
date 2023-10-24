import tensorflow as tf
import pandas as pd
from molgraph import layers
from molgraph.layers import MinMaxScaling
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.utils import shuffle
from helpers import RANDOM_STATE, upsample_minority, encoder


def make_graph_data(file, upsample=True, get_class_weights=True, debug=True):
    df_train = pd.read_csv(file)
    df_train = shuffle(
        df_train.reset_index(drop=True),
        random_state=RANDOM_STATE)

    if upsample:
        df_train = upsample_minority(df_train)

    if debug:
        print("Encoding the graphs, this might take a while...", flush=True)

    X,y = encoder(df_train.InChI.values), df_train.covalent.values

    if debug:
        print("Encoding complete!", flush=True)

    if get_class_weights:
        neg = len(df_train.query("covalent == 0"))
        pos = len(df_train.query("covalent == 1"))
        total = neg + pos
        weight_for_0 = (1 / neg) * (total / 2.0)
        weight_for_1 = (1 / pos) * (total / 2.0)
        class_weight = {0: weight_for_0, 1: weight_for_1}
        return X, y, class_weight

    return X, y


def train(X_train, y_train, class_weight,
          units=128, dropout=0.15, dense_units=256,
          activation="selu", learning_rate=5e-5, epochs=20, batch_size=64, verbosity=2):

    node_preprocessing = MinMaxScaling(
        feature='node_feature', feature_range=(0, 1), threshold=True)
    edge_preprocessing = MinMaxScaling(
        feature='edge_feature', feature_range=(0, 1), threshold=True)

    node_preprocessing.adapt(X_train.merge())
    edge_preprocessing.adapt(X_train.merge())

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(type_spec=X_train.unspecific_spec))
    model.add(node_preprocessing)
    model.add(edge_preprocessing)
    model.add(layers.GCNIIConv(units=units, activation=activation, dropout=dropout, use_edge_features=True))
    model.add(layers.GCNIIConv(units=units, activation=activation, dropout=dropout, use_edge_features=True))
    model.add(layers.Readout('mean'))
    model.add(tf.keras.layers.Dense(dense_units, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC(name="roc_auc")])

    model.fit(X_train, y_train,
                epochs=epochs,
                verbose=verbosity,
                batch_size=batch_size,
                class_weight=class_weight)
    return model


def get_test_metrics(test_file, model):
    X_test, y_test = make_graph_data(test_file, upsample=False, get_class_weights=False, debug=False)
    y_pred = model.predict(X_test)
    print(f"""
    Test AUC {roc_auc_score(y_test, y_pred)},
    Test Precision {precision_score(y_test, y_pred)},
    Test Recall {recall_score(y_test, y_pred)},
          """, flush=True)


def main():
    X_train, y_train, class_weight = make_graph_data("./data/train_test/test_data_all.csv", upsample=True, get_class_weights=True)
    model = train(X_train, y_train, class_weight)
    get_test_metrics("./data/train_test/test_data_all.csv", model)


if __name__ == "__main__":
    main()