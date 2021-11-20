# Configurations

- `ymls` example yaml config files.

## Introduction

<strong>TFCAIDM</strong> builds upon [JarvisMD](https://pypi.org/project/jarvis-md/) with additional yaml configuration files for model and training hyperparameters.

<strong>TFCAIDM</strong> supports training many different models under one train submission. Notice how all parameter values are enclosed in a list. Each elemenet in the list consists of a single training run. Figuring out how many models will be trained amounts to summing the amount of items in the list for each row and then multiplying each row. <strong>TFCAIDM</strong> will automatically generated every single possible permutation that is specified in the `model.yml` and `train.yml` files.

A full list of supported values are shown below.

---

## Templates

<details>
<summary>model.yml</summary>

```yaml
---
model:
  model: ["unet", "unet++", "unet3+"] # 3
  conv_type: ["conv"] # 1
  pool_type: ["conv", "max", "avg", "aspp", "acsp", "wasp"] # 6
  eblock: [
      "conv",
      "aspp",
      "ascp",
      "wasp",
      "cbam",
      "csp",
      "dense",
      "eca",
      "inception",
      "psp",
      "se",
      "u2net",
    ] # 12
  elayer: [1] # 1
  dblock: ["conv", "convgru", "attention"] # 3
  depth: [5] # 1
  width: [32] # 1
  width_scaling: [1.2] # 1
  kernel_size: [[1, 3, 3]] # 1
  strides: [[1, 2, 2]] # 1
  bneck: [2] # 1
  branches: [4] # 1
  atrous_rate: [6] # 1
  order: ["rnc", "nrc"] # 2
  norm: ["bnorm", "lnorm", "none"] # 3
  activ: ["leaky", "elu", "relu", "prelu", "gelu"] # 5
  attn_msk: ["softmax", "sigmoid", "tanh"] # 3
```

This configuration alone generates <strong>58,320</strong> different training runs.

</details>

<details>

<summary>train.yml</summary>

```yaml
# entires with `_id` can only contain a single value!
---
train:
  xs:
    dat:
      coord_id: ["coords"] # param coords must be present in client.yml!

  ys:
    lbl:
      mask_id: ["msk"] # param msk must be present in client.yml
      mask_weight: [1]
      output_weight: [5]
      head:
        [
          "encoder_classifier",
          "encoder_multi_scale_classifier",
          "decoder_classifier",
          "decoder_multi_scale_classifier",
          "decoder_deep_supervision_classifier",
          "decoder_complex_supervision_classifier",
        ]
      n_classes: [2]
      loss:
        [
          "sce",
          "wce",
          "mae",
          "mse",
          "dice",
          "logcosh_dice",
          "tversky",
          "focal_tversky",
          "focal",
        ]
      metric: ["acc", "bacc", "mae", "mse", "dice"]

  trainer:
    seed: [0]
    n_folds: [1]
    batch_size: [8]
    iters: [10000]
    steps: [1000]
    valid_freq: [5]
    lr: [1e-3]
    lr_alpha: [0.25] # range [0, 1]
    lr_decay: [0.97] # range (0, 1]
```

This configuration alone generates <strong>270</strong> different training runs.

If we consider that this `train.yml` was used in conjunction with the above `model.yml`, we would have <strong>15,746,400</strong> training runs in total! Obviously this is not feasible and should never be done...

</details>

---

## Suggestions

A methodology that can be used to guide the hyperparameter process.

<details>
<summary>Methodology</summary>

1. <strong>Figure out a good learning rate</strong>: train a model on different learning rates for a few epochs and use the learning rate that provides the lowest loss.
2. <strong>Find good hyperparameters for a single model type</strong>: hyperparameter search a given model until you get a strong score.
3. <strong>Find good hyperparameters for all model types</strong>: use the top-k best hyperpameters you found on a single model and train other model types using those hyperparameters.

Step 2. will take the longest

<strong>References</strong>

- [`A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, momentum, and weight decay`](https://arxiv.org/abs/1803.09820)

</details>
