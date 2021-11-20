"""This test case will validate tfcaidm imports!"""

from config import YAML_PATH


def test_imports():
    import tfcaidm

    from tfcaidm import Dataset, JClient, Jobs, Model, Loss, Metric, Trainer
    from tfcaidm.data import class_weights, positional_encoding
    from tfcaidm.jobs import config, env, params, tool
    from tfcaidm.losses import (
        dice,
        distance,
        entropy,
        focal,
        tversky,
        registry as loss_registry,
    )
    from tfcaidm.metrics import (
        acc,
        dice,
        distance,
        registry as metric_registry,
    )
    from tfcaidm.models import (
        audit,
        head,
        tasks,
        viz,
        registry as model_registry,
    )
    from tfcaidm.train import Summary, registry as train_registry

    assert True
