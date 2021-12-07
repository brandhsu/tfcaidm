import numpy as np
import tensorflow as tf
from tensorflow import optimizers
from tensorflow.keras import Input, layers, metrics

from jarvis.train import params
from jarvis.utils.general import gpus, overload

from tfcaidm import JClient
from tfcaidm import Model
from tfcaidm import Trainer
from tfcaidm.models import head
from tfcaidm.models import registry


# --- Autoselect GPU (use only on caidm cluster)
gpus.autoselect()

# --- Get hyperparameters
hyperparams = params.load()
fold = 0

# --- Modify dataset generator (applies same modification to train and valid generators)
@overload(JClient)
def create_generator(self, gen_data):
    for x, y in gen_data:

        """Each training pass will consist of N-1 contrastive comparisons"""
        """Note using custom layers so inputs and outputs are stored in x"""

        # --- Prepare ground-truths
        xs = x["dat"]
        ys = x["lbl"]
        assert (
            xs.shape[0] > 1 and ys.shape[0] > 1
        ), "ERROR! Batch size (N) must be at least 2!"

        xs_unk = xs[:-1]
        ys_unk = ys[:-1]
        xs_anc = np.stack([xs[-1]] * len(xs_unk))
        ys_anc = np.stack([ys[-1]] * len(ys_unk))

        # --- Assign ground-truths
        xs = {}
        ys = {}

        xs["anc"] = xs_anc
        xs["unk"] = xs_unk
        ys["ctr"] = tf.cast((ys_anc == ys_unk), tf.float32)
        ys["cls_anc"] = ys_anc
        ys["cls_unk"] = ys_unk

        yield xs, ys


# --- Custom loss functions
def cosine_similarity(vects):
    a, b = vects
    cosim = layers.Dot(axes=-1, normalize=True)([a, b])
    cosim = layers.Reshape((1, 1, 1, 1))(cosim)

    return 1 - cosim


def contrastive_loss(margin=1):
    def ctr_loss(y_true, y_pred):
        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum((margin - y_pred), 0))
        return tf.math.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)

    return ctr_loss


# --- Modify model
def AE(INPUT_SHAPE, hyperparams):
    autoencoder = registry.available_models()["ae"]

    # --- Extract hyperparams
    n = hyperparams["model"]["depth"]
    c = hyperparams["model"]["width"]
    k = hyperparams["model"]["kernel_size"]
    s = hyperparams["model"]["strides"]

    features = autoencoder(INPUT_SHAPE, n, c, k, s, hyperparams)
    embed = head.Encoder.last_layer(**features)

    # --- NOTE: The below hyperparameters will not be logged!
    conv = lambda filters, name: layers.Conv3D(
        filters=filters,
        kernel_size=1,
        activation="sigmoid",
        name=name,
        padding="same",
    )

    ftr = layers.GlobalAveragePooling3D()(embed)
    ftr = layers.Reshape((1, 1, 1, embed.shape[-1]))(ftr)
    ctr = conv(filters=10, name="ctr")(ftr)
    cls = conv(filters=1, name="cls")(ctr)

    logits = {}
    logits["ctr"] = ctr
    logits["cls"] = cls

    return logits


@overload(Model)
def create(self):

    # --- User defined code
    INPUT_SHAPE = (96, 160, 160, 1)
    inputs = Input(shape=INPUT_SHAPE)
    outputs = AE(inputs, self.hyperparams)

    # --- Create tensorflow model
    backbone = self.assemble(inputs=inputs, outputs=outputs)

    inputs = {
        "anc": Input(shape=INPUT_SHAPE, name="anc"),
        "unk": Input(shape=INPUT_SHAPE, name="unk"),
    }

    # --- Define contrastive network
    anc_net = backbone(inputs=inputs["anc"])
    unk_net = backbone(inputs=inputs["unk"])

    # --- Cosine similarity embeddings
    ctr = layers.Lambda(cosine_similarity)([anc_net["ctr"], unk_net["ctr"]])

    logits = {}
    logits["ctr"] = layers.Layer(name="ctr")(ctr)
    logits["cls_anc"] = layers.Layer(name="cls_anc")(anc_net["cls"])
    logits["cls_unk"] = layers.Layer(name="cls_unk")(unk_net["cls"])

    # --- Create tensorflow model
    model = self.assemble(inputs=inputs, outputs=logits)

    model.compile(
        optimizer=optimizers.Adam(
            learning_rate=self.hyperparams["train"]["trainer"]["lr"]
        ),
        loss={"ctr": contrastive_loss()},
        metrics={
            "cls_anc": metrics.BinaryAccuracy(),
            "cls_unk": metrics.BinaryAccuracy(),
        },
        experimental_run_tf_function=False,
    )

    return model


# --- Train model (dataset and model created within trainer)
trainer = Trainer(hyperparams)
results = trainer.cross_validation(save=False)
trainer.save_results(results)
