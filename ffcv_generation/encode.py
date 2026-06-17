from ffcv.writer import DatasetWriter
from ffcv.fields import NDArrayField, JSONField
import wandb
import argparse
import os
import json
import random
import numpy as np
import torch
from feature_dataset import classification_dataset

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--config", "-c", default="config.yaml", help="wandb config file.")
parser.add_argument("--split", type=str, help='Specify split to encode from the config file.')
parser.add_argument("--holdout_domain", type=str, default='iiith', help='Specify holdout OOD domain')

args = parser.parse_args()

set_seed()
wandb.init(config=args.config, mode="disabled")
config = wandb.config
feat_dim = config.feat_dim
n_labels = len(config.labels)
n_temporal_samples = 3

annotations_path = os.path.join(config.repo_root, 'annotations')
ffcv_path        = os.path.join(config.repo_root, 'betons')

out_fn = f'{args.holdout_domain}_Ego4OOD_{args.split}.beton'
out_path = os.path.join(ffcv_path, out_fn)

moment2index   = json.load(open(os.path.join(annotations_path, 'moment_to_idx.json')))
category2index = json.load(open(os.path.join(annotations_path, 'category_to_idx.json')))
domain2index   = json.load(open(os.path.join(annotations_path, 'domain_to_idx.json')))

if args.split == 'train':
    ds = classification_dataset(config, 'train', moment2index, category2index, domain2index, holdout_domain=args.holdout_domain, ood=False)
elif args.split == 'test':
    ds = classification_dataset(config, 'train', moment2index, category2index, domain2index, holdout_domain=args.holdout_domain, ood=True)
elif args.split == 'val':
    ds = classification_dataset(config, 'val', moment2index, category2index, domain2index, holdout_domain=args.holdout_domain, ood=False)
else:
    raise NotImplementedError(f"Unknown split: {args.split}")

print('Writing FFCV to: ', out_path)

writer = DatasetWriter(
    out_path,
    {
        'info':  JSONField(),
        'feat':  NDArrayField(shape=(n_temporal_samples, feat_dim), dtype=np.dtype('float32')),
        'label': NDArrayField(shape=(n_labels,), dtype=np.dtype('int32'))
    },
    num_workers=config.n_workers
)

writer.from_indexed_dataset(ds, shuffle_indices=config.ffcv_pre_shuffle)
print('Done writing FFCV file.')
wandb.finish()
