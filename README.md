# ML-Assignment-2
Machine Learning assignment 2 at USI, Lugano

## Run the model(s)

Be sure to be in `deliverable/`; `custom_utils.py` is required to be in the same folder than the script!

```console
python run_task1.py
```

or 

```console
python run_task2.py
```

For task 2, the script will automatically download the rps dataset if not present.


## Compile and train the models(s)

Be sure to be in `src/`; `custom_utils.py` is required to be in the same folder than the script!

You can either specify a task to compile only that model or no arguments to compile both. This will create both the `.png` plots, the `.h5` and the `.pkl` models.

```console
python create_models.py [num_task]
```
