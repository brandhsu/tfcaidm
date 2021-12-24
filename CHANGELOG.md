# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

## [0.0.0a5] - 2021-12-24

### Added

- Add model benchmarking to the caidm clusters, site available at [tfcaidm-site](https://brandhsu.github.io/tfcaidm-site/)
- Add helper functions in [tfcaidm.model](https://github.com/Brandhsu/tfcaidm/blob/master/tfcaidm/models/model.py) for streamlined model inference

### Deprecated

- Deprecated method `create_generator` in [tfcaidm.JClient](https://github.com/Brandhsu/tfcaidm/blob/master/tfcaidm/data/jclient.py) in favor of distinct generators for training (`model.fit`) and model evaluation (`model.eval`). These methods are now called `train_generator` and `eval_generator`, for any projects `@overloading create_generator` please make the necessary project changes. The default behavior of `eval_generator` is the same as `train_generator`.

## [0.0.0a4] - 2021-12-12

### Added

- Add tensorboard hyperparameter logging
- Add identifiable loss names during training
- Add model inference method for filtering inputs and outputs

## [0.0.0a3] - 2021-11-23

### Added

- Support for more python3 versions

## [0.0.0a2] - 2021-11-20

### Fixed

- **TFCAIDM** logo in `README.md`

## [0.0.0a1] - 2021-11-20

### Added

- Initial release 🎉
