## Models

<strong>TFCAIDM</strong> offers various model architectures to use.

<details>
<summary>Available Models</summary>

All models are configurable and interchangeable, i.e. hyperparameters are easily tunable (number of layers, number of channels, growth factor, pooling method, etc.) and different layer configurations are also available for all of the implemented model architectures.

<strong>Architectures Implemented</strong>

_Hyperlinks linked to arxiv._

- [x] [Deep CNN](https://arxiv.org/abs/2004.02806) (survey paper)
- [x] [ResNet](https://arxiv.org/abs/1512.03385)
- [x] [UNet](https://arxiv.org/abs/1505.04597)
- [x] [UNet++](https://arxiv.org/abs/1807.10165)
- [x] [UNet3+](https://arxiv.org/abs/2004.08790)

<strong>Layers Implemented</strong>

- [x] [ASPP: Atrous Spatial Pyramid Pooling](https://arxiv.org/abs/1606.00915v2)
- [x] [Attention Gate](https://arxiv.org/abs/1804.03999)
- [x] [DenseNet](https://arxiv.org/abs/1608.06993)
- [x] [CSPNet](https://arxiv.org/abs/1911.11929)
- [x] [CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521)
- [x] [ECA-Net: Efficient Channel Attention for Deep CNNs](https://arxiv.org/abs/1910.03151)
- [x] [Inception: Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)
- [x] [PSPNet: Pyramid Scene Parsing Network](https://arxiv.org/abs/1612.01105)
- [x] [SE-Net: Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)
- [x] [U2-Net: Going Deeper with Nested U-Structure for Salient Object Detection](https://arxiv.org/abs/2005.09007)
- [x] [Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357) (currently experimental, performance is not great)
- [x] [CoordConv: An Intriguing Failing of CNNs](https://arxiv.org/abs/1807.03247) (applied to the input layer only)

</details>

<details>
<summary>Configuring Models</summary>

All models are configurable through a yaml file. See [`configs`](https://github.com/Brandhsu/tfcaidm-pkg/tree/main/configs) for more information, namely yamls with the name `pipeline.yml` (note: the name is just a convention and can be changed).

<strong>Input blocks aka.</strong> `iblock`

These blocks are placed in the first layer of the network. A network will use only one specific `iblock` per creation.

```python
# --- Available blocks:

iblock = {
    'none':       # Raw Input
    'coord':      # concat( CoordConv ; Raw Input )
}

```

<strong>Encoder blocks aka.</strong> `eblock`

These blocks are placed in the encoder portion of the network. A network will use only one specific `eblock` per creation.

```python
# --- Available blocks:

eblock = {
    'aspp':      # Atrous Spatial Pyramid Pooling
    'acsp':      # Atrous Cascaded Spatial Pooling
    'wasp':      # Waterfall Atrous Spatial Pooling
    'cbam':      # Convolutional Block Attention Module
    'conv':      # Convolutional Neural Network
    'csp':       # A New Backbone that can Enhance Learning Capability of CNN
    'dense':     # Densely Connected Convolutional Networks
    'eca':       # Efficient Channel Attention for Deep Convolutional Neural Networks
    'inception'  # Going Deeper with Convolutions
    'psp':       # Pyramid Scene Parsing Network
    'se':        # Squeeze-and-Excitation Networks
    'u2net':     # Going Deeper with Nested U-Structure for Salient Object Detection
}

```

<strong>Decoder blocks aka.</strong> `dblock`

These blocks are placed in the decoder portion of the network. A network will use only one specific `dblock` per creation.

```python
# --- Available blocks:

dblock = {
    'attention': # Attention U-Net
    'convgru':   # Convolutional Gated Recurrent Network
    'conv':      # Transposed Convolutional Neural Network
}

```

<strong>Pooling layers aka.</strong> `pool_type`

These pooling layers are used throughout the network. A network will use only one specific `pool_type` per creation.

```python
# --- Available elayer:

pool_type = {
    'max':       # Max Pooling
    'avg':       # Average Pooling
    'conv':      # Strided Convolution
    'aspp':      # Atrous Spatial Pyramid Pooling
    'acsp':      # Atrous Cascaded Spatial Pooling
    'wasp':      # Waterfall Atrous Spatial Pooling
}

```

<strong>Aggregation layers aka.</strong> `agg_type`

These aggregation layers are used when combining encoder and decoder features in a UNet. A network will use only one specific `agg_type` per creation.

NOTE: this is not implemented yet!

```python
# --- Available elayer:

agg_type = {
    'add':       # Add features
    'mean':      # Average features
    'concat':    # Concat features
}

```

</details>

<details>
<summary>Model Outputs</summary>

When accessing an automatically generated model through:

```python
from tfcaidm.models import registry
from tfcaidm.models import head

backbone = registry.custom_mode()
outputs = backbone(**hyperparams["model"]) # feed model args
encoder = head.Encoder.last_layer(**outputs) # get last encoder layer
```

Var `outputs` will contain a dictionary of layers `{"encoders", "decoder", "full_res"}` which correspond to the post-activation outputs of a model block. Var `encoder` will contain the last encoder block output using some helper functions defined in the `head` module.

</details>
