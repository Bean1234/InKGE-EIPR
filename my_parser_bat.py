import argparse
import json
import logging
import os
import sys


def parse(test=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default = "./data/", type = str)
    parser.add_argument('--data_name', default = 'NL-25', type = str)
    parser.add_argument('--exp', default = 'exp_relgraph', type = str)
    parser.add_argument('-m', '--margin', default = 8, type = float)
    parser.add_argument('-lr', '--learning_rate', default=0.00006, type = float)
    parser.add_argument('-nle', '--num_layer_ent', default = 2, type = int)
    parser.add_argument('-nlr', '--num_layer_rel', default = 2, type = int)
    parser.add_argument('-d_e', '--dimension_entity', default = 128, type = int)
    parser.add_argument('-d_r', '--dimension_relation', default = 128, type = int)
    parser.add_argument('-hdr_e', '--hidden_dimension_ratio_entity', default = 8, type = int)
    parser.add_argument('-hdr_r', '--hidden_dimension_ratio_relation', default = 4, type = int)
    parser.add_argument('-b', '--num_bin', default = 10, type = int)
    parser.add_argument('-e', '--num_epoch', default = 15000, type = int)
    # parser.add_argument('--full_graph_neg', default="True", help="Use full graph negative sampling for nodes")
    if test:
        parser.add_argument('--target_epoch', default = 15000, type = int)
    parser.add_argument('-v', '--validation_epoch', default = 200, type = int)
    
    parser.add_argument('--num_head', default = 8, type = int)
    parser.add_argument('--num_neg', default = 10, type = int)
    parser.add_argument('--best', action = 'store_true')
    if not test:
        parser.add_argument('--no_write', action = 'store_true')

    args = parser.parse_args()


    if not test:
        os.makedirs(f"./ckpt/{args.exp}/{args.data_name}", exist_ok=True)
    file_format = f"lr_{args.learning_rate}_dim_{args.dimension_entity}_{args.dimension_relation}" + \
                  f"_bin_{args.num_bin}_total_{args.num_epoch}_every_{args.validation_epoch}" + \
                  f"_neg_{args.num_neg}_layer_{args.num_layer_ent}_{args.num_layer_rel}" + \
                  f"_hid_{args.hidden_dimension_ratio_entity}_{args.hidden_dimension_ratio_relation}" + \
                  f"_head_{args.num_head}_margin_{args.margin}"
    logger = logging.getLogger()
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
    console_formatter = logging.Formatter('%(asctime)-8s: %(message)s')
    log_path = f"ckpt/{args.exp}/{args.data_name}"
    logging_file_name = f"{log_path}/log_{file_format}.log"
    file_handler = logging.FileHandler(logging_file_name)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.formatter = console_formatter
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    args.logger = logger

    if test and args.best:
        remaining_args = []
        with open(f"./ckpt/best/{args.data_name}/config.json") as f:
            configs = json.load(f)
        for key in vars(args).keys():
            if key in configs:
                vars(args)[key] = configs[key]
            else:
                remaining_args.append(key)

    return args