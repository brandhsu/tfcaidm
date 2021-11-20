"""Model head interface"""

from tfcaidm.models.layers import learner
from tfcaidm.models.layers import transform


class Decoder:
    """Determines the decoder head to use for a given output"""

    @classmethod
    def last_layer(cls, decoders, *args, **kwargs):
        return decoders[-1]

    @classmethod
    def multi_scale(cls, decoders, *args, **kwargs):
        # Assumes 5D tensor of [batch, depth, width, height, channels]
        layers = []

        final_shape = decoders[-1].shape
        f_d, f_w, f_h, f_c = final_shape[1:]

        for decoder in decoders:
            inter_shape = decoder.shape
            i_d, i_w, i_h = inter_shape[1:-1]

            s = (int(f_d / i_d), int(f_w / i_w), int(f_h / i_h))

            decoder = learner.pre_activation_conv(x=decoder, c=f_c, k=1)

            layer = transform.up_sample(decoder, s=s)
            layers.append(layer)

        layer = transform.add(layers)

        return layer

    @classmethod
    def deep_supervision(cls, full_res, *args, **kwargs):
        layer = transform.add(full_res)
        return layer

    @classmethod
    def complex_supervision(cls, full_res, decoders, *args, **kwargs):
        decoders = cls.multi_scale(decoders)
        full_res = cls.deep_supervision(full_res)
        layer = decoders + full_res
        return layer


class Encoder:
    """Determines the encoder head to use for a given output"""

    @classmethod
    def last_layer(cls, encoders, *args, **kwargs):
        return encoders[-1]

    @classmethod
    def multi_scale(cls, encoders, *args, **kwargs):
        # Assumes 5D tensor of [batch, depth, width, height, channels]
        layers = []

        final_shape = encoders[-1].shape
        f_d, f_w, f_h = final_shape[1:-1]
        f_c = encoders[0].shape[-1]

        for encoder in encoders:
            inter_shape = encoder.shape
            i_d, i_w, i_h = inter_shape[1:-1]

            s = (int(i_d / f_d), int(i_w / f_w), int(i_h / f_h))

            encoder = learner.pre_activation_conv(x=encoder, c=f_c, k=1)

            layer = transform.average_pool(encoder, s=s) * 0.5
            layer += transform.max_pool(encoder, s=s) * 0.5
            layers.append(layer)

        layer = transform.add(layers)

        return layer
