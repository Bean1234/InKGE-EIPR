# Submit to CIKM 2024

## Requirements

We used Python 3.8 and PyTorch 1.12.1 with cudatoolkit 11.3.

You can install all requirements with:

```shell
--extra-index-url https://download.pytorch.org/whl/cu113
numpy==1.23.4
python-igraph==0.10.2
tqdm==4.64.1
torch==1.12.1+cu113
scipy==1.9.3pip install -r requirements.txt
```

## Training and evaluation
1. Train model
To train model from scratch, run `train_bat.py` with arguments. Please refer to `my_parser_bat.py` for the examples of the arguments. Please tune the hyperparameters of our model using the range provided in paper because the best hyperparameters may be different due to randomness.

We used NVIDIA GeForce RTX 4090 for all our experiments.

The list of arguments of `train.py`:

- `--data_name`: name of the dataset
- `--exp`: experiment name
- `-m, --margin`
- `-lr, --learning_rate`
- `-nle, --num_layer_ent`
- `-nlr, --num_layer_rel`
- `-d_e, --dimension_entity`
- `-d_r, --dimension_relation`
- `-hdr_e, --hidden_dimension_ratio_entity`
- `-hdr_r, --hidden_dimension_ratio_relation`
- `-b, --num_bin`
- `-e, --num_epoch`
- `--target_epoch`: the epoch to run test (only used for test.py)
- `-v, --validation_epoch`: duration for the validation
- `--num_head`
- `--num_neg`: number of negative triplets per triplet
- `--best`: use the provided checkpoints (only used for test.py)
- `--no_write`: don't save the checkpoints (only used for train.py)

2. Evaluate model

```
python test_bat.py --data_name <dataset-name> --exp <experiment name> --target_epoch <target_epoch>
```
## Acknowledgement

We refer to the code of [INGRAM](https://github.com/bdi-lab/InGram/tree/main). Thanks for their contributions.
