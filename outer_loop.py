import numpy as np
import pandas as pd
import time
import Models
import pickle

from scipy.stats import sem
from evaluation import Evaluator

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--model', default='SLearner', type=str, help='model type (default: SLearner)')
parser.add_argument('--data', default='IHDP', type=str, help='dataset (default: IHDP')
args = parser.parse_args()

train = dict(np.load('Data/%s/train.npz' % args.data))

if np.unique(train['yf']).tolist() == [0, 1]:
	binary = True
else:
	binary = False

model = Models.__dict__[args.model](binary=binary)

replications = train['x'].shape[2]
if args.model == 'OrthoForest':
    print('OrthoForest in use')
    replications = 1
scores_in = []
scores_out = []

for repl in range(replications):
	train_x = train['x'][:, :, repl]
	train_t = train['t'][:, repl]
	train_y = train['yf'][:, repl]

	start = time.time()
	model.fit(train_x, train_t, train_y)
	end = time.time()
	train_time = end - start

	## In-sample evaluation:
	start = time.time()
	train_preds0 = model.predict(train_x, np.zeros(train_t.shape))
	train_preds1 = model.predict(train_x, np.ones(train_t.shape))
	end = time.time()
	in_sample_time = end - start

	## In-sample:
	eval_in = Evaluator(
		y = train_y, 
		t = train_t.flatten(),
		y_cf = train['ycf'][:, repl] if 'ycf' in train else None,
		mu0 = train['mu0'][:, repl] if 'mu0' in train else None,
		mu1 = train['mu1'][:, repl] if 'mu1' in train else None,
		true_ite = train['ite'][:, repl] if 'ite' in train else None,
		true_ate = train['ate'].item() if 'ate' in train else None
	)

	stats_names, in_sample_scores = eval_in.calc_stats(train_preds1, train_preds0)
	stats_names.append('TRAIN_TIME')
	stats_names.append('PREDICT_TIME')
	scores_in.append(np.concatenate([in_sample_scores, [train_time, in_sample_time]]))

	test = dict(np.load('Data/%s/test.npz' % args.data))
	test_x = test['x'][:, :, repl]
	test_t = test['t'][:, repl]
	test_y = test['yf'][:, repl]

	## Out-of-sample:
	start = time.time()
	test_preds0 = model.predict(test_x, np.zeros(test_t.shape))
	test_preds1 = model.predict(test_x, np.ones(test_t.shape))
	end = time.time()
	out_of_sample_time = end - start

	## Out-of-sample:
	eval_out = Evaluator(
		y = test_y, 
		t = test_t,
		y_cf = test['ycf'][:, repl] if 'ycf' in test else None,
		mu0 = test['mu0'][:, repl] if 'mu0' in test else None,
		mu1 = test['mu1'][:, repl] if 'mu1' in test else None,
		true_ite = test['ite'][:, repl] if 'ite' in test else None,
		true_ate = test['ate'].item() if 'ate' in test else None
	)

	stats_names, out_of_sample_scores = eval_out.calc_stats(test_preds1, test_preds0)
	stats_names.append('TRAIN_TIME')
	stats_names.append('PREDICT_TIME')
	scores_out.append(np.concatenate([out_of_sample_scores, [train_time, out_of_sample_time]]))

scores_in = np.vstack(scores_in)
scores_out = np.vstack(scores_out)
means_in, stds_in = np.mean(scores_in, axis=0), sem(scores_in, axis=0)
print ('In sample')
print (means_in)
print (stds_in)

means_out, stds_out = np.mean(scores_out, axis=0), sem(scores_out, axis=0)
print ('Out of sample')
print (means_out)
print (stds_out)

idxs = ['IN_MEAN', 'IN_STD', 'OUT_MEAN', 'OUT_STD']
df = pd.DataFrame(np.vstack([means_in, stds_in, means_out, stds_out]), index=idxs, columns=stats_names)
df.to_csv('Results/' + args.model + '_' + args.data + '.csv')




