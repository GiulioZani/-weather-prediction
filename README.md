Run:
```
python -m main <model-name> <{train, test}>
```
Example:
```
python -m main simple_regression train
```
Models are located in the `model` folder. See `simple_regression` for details.

All taining hyperparameters are located in `training_parameters.json`. Every parameter can be overwritten by `--<param-name>` option via CLI.

