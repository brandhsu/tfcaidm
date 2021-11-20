"""Select model blocks to use"""

import tfcaidm.models.custom.registry as registry
import tfcaidm.models.layers.transform as transform
import tfcaidm.common.constants as constants

# --- conv_type in csv
def conv_selection(
    x=None, c=1, k=(1, 1, 1), s=(1, 1, 1), d=(1, 1, 1), name=None, hyperparams={}
):
    """Choose between different types of convolution methods"""

    conv_types = registry.available_convs()

    # --- Extract hyperparams from params csv
    conv_type = hyperparams["model"]["conv_type"]

    if conv_type not in conv_types:
        raise ValueError(f"ERROR! Conv layer `{conv_type}` is not defined!")

    return conv_types[conv_type](
        x=x, c=c, k=k, s=s, d=d, name=name, hyperparams=hyperparams
    )


# --- conv_type in csv (same settings as conv_selection)
def tran_selection(
    x=None, c=1, k=(1, 1, 1), s=(1, 2, 2), d=(1, 1, 1), name=None, hyperparams={}
):
    """Choose between different types of transpose convolution methods"""

    conv_types = registry.available_trans()

    # --- Extract hyperparams from params csv
    conv_type = hyperparams["model"]["conv_type"]

    if conv_type not in conv_types:
        raise ValueError(f"ERROR! Conv layer `{conv_type}` is not defined!")

    return conv_types[conv_type](
        x=x, c=c, k=k, s=s, d=d, name=name, hyperparams=hyperparams
    )


# ---pool_type in csv
def pool_selection(x, c, k, s, hyperparams):
    """Choose between different pooling layers"""

    pool_types = registry.available_pools()

    # --- Extract hyperparams from params csv
    pool_type = hyperparams["model"]["pool_type"]

    if pool_type not in pool_types:
        raise ValueError(f"ERROR! Pool layer `{pool_type}` is not defined!")

    return pool_types[pool_type](x=x, c=c, k=k, s=s, hyperparams=hyperparams)


# --- eblock in csv
def encoder_selection(x, c, k, s, hyperparams):
    """Choose between different encoder feature extraction blocks"""

    eblocks = registry.available_encoders()

    # --- Extract hyperparams from params csv
    eblock = hyperparams["model"]["eblock"].split("_")[0]

    if eblock not in eblocks:
        raise ValueError(f"ERROR! Encoder block `{eblock}` is not defined!")

    return eblocks[eblock](x=x, c=c, k=k, s=s, hyperparams=hyperparams)


# --- dblock in csv
def decoder_selection(x, x_skip, c, k, s, hyperparams):
    """Choose between different decoder feature extraction blocks"""

    dblocks = registry.available_decoders()

    # --- Extract hyperparams from params csv
    dblock = hyperparams["model"]["dblock"]

    if dblock not in dblocks:
        raise ValueError(f"ERROR! Decoder block `{dblock}` is not defined!")

    return dblocks[dblock](
        x=x,
        x_skip=x_skip,
        c=c,
        k=k,
        s=s,
        hyperparams=hyperparams,
    )


def head_selection(x, hyperparams):
    """Choose between different output heads"""

    heads = registry.available_heads()

    # --- Extract hyperparams from params csv
    head = hyperparams["head"]

    if head not in heads:
        raise ValueError(f"ERROR! Model head `{head}` is not defined!")

    return heads[head](**x)


def task_selection(x, inputs, client, task="auto"):
    """Choose between different task(s)"""

    task_types = registry.available_tasks()

    if task not in task_types:
        raise ValueError(f"ERROR! Model tasks `{task}` is not defined!")

    return task_types[task](x, inputs, client)


# --- model in csv
def model_selection(x, hyperparams):
    """Choose between different model architectures"""

    model_archs = registry.available_models()

    # --- Extract hyperparams from params csv
    model_arch = hyperparams["model"]["model"]
    n = hyperparams["model"]["depth"]
    c = hyperparams["model"]["width"]
    k = hyperparams["model"]["kernel_size"]
    s = hyperparams["model"]["strides"]

    if model_arch not in model_archs:
        raise ValueError(f"ERROR! Model `{model_arch}` is not defined!")

    return model_archs[model_arch](x, n, c, k, s, hyperparams)


def input_selection(x, name, hyperparams):
    """Concatenate and project additional input features"""

    # --- Choose first input when not provided
    if name is None:
        names = [*hyperparams["train"]["xs"].keys()]
        assert len(names), "ERROR! Issue building inputs, check your client yaml file!"
        name = names[0]

    # --- Extract hyperparams from params csv
    addons = hyperparams["train"]["xs"][name]
    c = hyperparams["model"]["width"]

    features = [name]

    if type(addons) == dict:
        for k in addons:
            if type(addons[k]) == dict:
                features += [addons[k][n] for n in addons[k] if n == "name"]

    x_new = [x[i] for i in x if i in features]
    x_new = transform.concat(x_new)

    return conv_selection(x=x_new, c=c, k=1, s=1, hyperparams=hyperparams)
