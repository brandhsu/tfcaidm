"""Model visualization tools"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

import tensorflow as tf
from tensorflow.keras import activations, models
from jarvis.utils.display import montage


class Base:
    def __init__(self, model, **kwargs):
        self.model = model

    def get_layer(self, layer):
        if type(layer) == int:
            hidden = self.model.layers[layer]
        elif type(layer) == str:
            hidden = self.model.get_layer(layer)
        else:
            raise ValueError(
                "ERROR! `layer` must be either a `string` or `int` denoting the layer name or layer index respectively."
            )

        return hidden

    def get_grad_model(self, layer, output_name):
        grad_model = models.Model(
            [self.model.inputs],
            {
                "hidden": self.get_layer(layer).output,
                "output": self.model.output[output_name],
            },
        )

        return grad_model

    def norm(self, x):
        return activations.relu(x) / tf.reduce_max(x, keepdims=True)

    def heatmap(self, x, layer, output_name, class_of_interest=1, **kwargs):
        grad_cam = self.get_grad_cam(x, layer, output_name, class_of_interest)
        heat_map = self.norm(grad_cam)

        return heat_map


class GradCAM(Base):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)

    def get_grad_cam(self, x, layer, output_name, class_of_interest=1, **kwargs):
        grad_model = self.get_grad_model(layer, output_name)

        with tf.GradientTape() as tape:
            model_outputs = grad_model(x)
            hidden = model_outputs["hidden"]
            output = model_outputs["output"]
            cam = output[..., class_of_interest]

        grad = tape.gradient(cam, hidden)
        axis = [*range(len(grad.shape[:-1]))]
        pool_grad = tf.reduce_mean(grad, axis=axis)
        grad_cam = hidden @ pool_grad[..., tf.newaxis]

        return grad_cam


class GradCAMpp(Base):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)

    def get_grad_cam(self, x, layer, output_name, class_of_interest=1, **kwargs):
        grad_model = self.get_grad_model(layer, output_name)

        with tf.GradientTape() as tape:
            model_outputs = grad_model(x)
            hidden = model_outputs["hidden"]
            output = model_outputs["output"]
            cam = output[..., class_of_interest]

        grad = tape.gradient(cam, hidden)
        grad = activations.relu(grad) / grad
        axis = [*range(len(grad.shape[:-1]))]
        pool_grad = tf.reduce_sum(grad, axis=axis)
        grad_cam = hidden @ pool_grad[..., tf.newaxis]

        return grad_cam


def overlay(x, heatmap, cmap="jet", alpha=0.2, figsize=(7, 7)):

    # --- Use Jarvis montage function to collapse into a N x N grid
    im = np.squeeze(montage(x))
    hm = np.squeeze(montage(heatmap))

    # --- Zoom
    hm = zoom(hm, zoom=np.array(im.shape) / np.array(hm.shape), order=1)

    # --- Draw figure
    plt.clf()
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(im, cmap="gray")
    plt.imshow(hm, cmap=cmap, alpha=alpha)
