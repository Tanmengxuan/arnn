# arnn


## Train the model

To train the model, run

```
$ python main.py --train --model_name <name of your model> 
```

This trains a model with default parameters:

- Training epochs: `--epoch 150`
- Mini batch size: `--batch_size 64`
- GRU units: `--rnn 20`
- Length of sequence for training: `--len 40`
- Dropout rate: `--drop 0.001`


## Test the model

#### Test on a specific sequence 

To test on a specific sequence in the test data and outputting a prediction sequence of your choice,
you can use the `--test_sample` and `--test_steps` flags.

For instance, running the command
```
$ python main.py --test --model_name <name of your model> --test_sample 0:30 --test_steps 10
```
will feed the model preprocessed samples from row index 0 to 30 as an input sequence and the it will predict the 
values that are 10 steps ahead. 

#### Visualize the predicted values 

`--test_sample` is usually used with `--plot_sample` to visualize the predicted samples against the target values.

```
$ python main.py --test --model_name <name of your model> --test_sample 0:30 --test_steps 10 --plot_sample
```

