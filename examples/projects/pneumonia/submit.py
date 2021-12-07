from tfcaidm import Jobs

YML_CONFIG = "/home/brandon/tfcaidm-pkg/configs/ymls/xr_pna/pipeline.yml"
TRAIN_ROUTINE_PATH = "main.py"

Jobs(path=YML_CONFIG).setup(
    producer=__file__,
    consumer=TRAIN_ROUTINE_PATH,
).train_cluster(gpu="rtx|titan")
