import sys

from relgraph_bat import generate_relation_triplets
from dataset import TrainData, TestNewData
from tqdm import tqdm
import random
from model_bat import InGram
import torch
import numpy as np
from utils import generate_neg
import os
from evaluation_old_bat import evaluate
from initialize_bat import initialize
from my_parser_bat import parse
import logging

OMP_NUM_THREADS = 10
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True
torch.set_num_threads(8)
torch.cuda.empty_cache()
args = parse(False)
assert args.data_name in os.listdir(args.data_path), f"{args.data_name} Not Found"
path = args.data_path + args.data_name + "/"
train = TrainData(path)
valid = TestNewData(path, data_type = "valid")


file_format = f"lr_{args.learning_rate}_dim_{args.dimension_entity}_{args.dimension_relation}" + \
			  f"_bin_{args.num_bin}_total_{args.num_epoch}_every_{args.validation_epoch}" + \
			  f"_neg_{args.num_neg}_layer_{args.num_layer_ent}_{args.num_layer_rel}" + \
			  f"_hid_{args.hidden_dimension_ratio_entity}_{args.hidden_dimension_ratio_relation}" + \
			  f"_head_{args.num_head}_margin_{args.margin}"

d_e = args.dimension_entity
d_r = args.dimension_relation
hdr_e = args.hidden_dimension_ratio_entity
hdr_r = args.hidden_dimension_ratio_relation
B = args.num_bin
epochs = args.num_epoch
valid_epochs = args.validation_epoch
num_neg = args.num_neg

my_model = InGram(dim_ent = d_e, hid_dim_ratio_ent = hdr_e, dim_rel = d_r, hid_dim_ratio_rel = hdr_r, \
				num_bin = B, num_layer_ent = args.num_layer_ent, num_layer_rel = args.num_layer_rel, \
				num_head = args.num_head)
args.logger.info('File_format:{}'.format(file_format))

my_model = my_model.cuda()
loss_fn = torch.nn.MarginRankingLoss(margin = args.margin, reduction = 'mean')

optimizer = torch.optim.Adam(my_model.parameters(), lr = args.learning_rate)
pbar = tqdm(range(epochs))

total_loss = 0
best_val_mrr = 0
best_val_hit1 = 0
early_stop_counter = 0
best_epoch_mrr = 0
best_epoch_hit1 =0
BEST_METRIC_KEY = 'MRR'
args.logger.info('BEST_METRIC_KEY:{}'.format(BEST_METRIC_KEY))


for epoch in pbar:
	optimizer.zero_grad()
	p = random.choice([0.65,0.85,0.75])
	msg, sup = train.split_transductive(p)

	init_emb_ent, init_emb_rel, relation_triplets, rel_graph,G_nx = initialize(train, msg, d_e, d_r, B)
	msg = torch.tensor(msg).cuda()
	sup = torch.tensor(sup).cuda()

	emb_ent, emb_rel = my_model(init_emb_ent, init_emb_rel, msg, relation_triplets)
	pos_scores = my_model.score(emb_ent, emb_rel, sup,G_nx)
	neg_scores = my_model.score(emb_ent, emb_rel, generate_neg(sup, train.num_ent, num_neg = num_neg),G_nx)

	loss = loss_fn(pos_scores.repeat(num_neg), neg_scores, torch.ones_like(neg_scores))

	loss.backward()
	torch.nn.utils.clip_grad_norm_(my_model.parameters(), 0.16, error_if_nonfinite = False)
	optimizer.step()
	total_loss += loss.item()
	pbar.set_description(f"loss {loss.item()}")
	

	args.logger.info('Epoch:{}, total_loss:{}'.format(epoch,total_loss))

	if ((epoch + 1) % valid_epochs) == 0:
		print("Validation")
		my_model.eval()
		val_init_emb_ent, val_init_emb_rel, val_relation_triplets,val_rel_graph,G_valid= initialize(valid, valid.msg_triplets, d_e, d_r, B)

		metrics = evaluate(my_model, valid, epoch, val_init_emb_ent, val_init_emb_rel, val_relation_triplets,G_valid)

		if not args.no_write:
			torch.save({'model_state_dict': my_model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
						'inf_emb_ent': val_init_emb_ent, 'inf_emb_rel': val_init_emb_rel}, \
					   f"ckpt/{args.exp}/{args.data_name}/{file_format}_{epoch + 1}.ckpt")
		if metrics[BEST_METRIC_KEY] > best_val_mrr:
			best_val_mrr = metrics[BEST_METRIC_KEY]
			best_epoch_mrr = epoch
			args.logger.info('BestEpoch_MRR:{}, best_val_mrr:{:.3f}'.format(best_epoch_mrr, best_val_mrr))
		if metrics['Hits@1'] > best_val_hit1:
			best_val_hit1 = metrics['Hits@1']
			best_epoch_hit1 = epoch
			args.logger.info('BestEpoch_Hit1:{}, best_val_hit1:{:.3f}'.format(best_epoch_hit1, best_val_hit1))

		my_model.train()
args.logger.info('Final BestEpoch_MRR:{}, best_val_mrr:{:.3f}'.format(best_epoch_mrr, best_val_mrr))
args.logger.info('Final BestEpoch_Hit1:{}, best_val_hit1:{:.3f}'.format(best_epoch_hit1, best_val_hit1))

	
