import torch
import numpy as np
import pandas as pd
import os


def parse_featpath(p):
    directory, filename = os.path.split(p)
    name = filename[:-4]  
    uid, start, end, *cls = name.split("_")
    return pd.Series({
        "uid": uid,
        "start_frame": int(start),
        "end_frame": int(end),
        "moment": "_".join(cls)
    })

class classification_dataset(torch.utils.data.Dataset):
    """
    Use for extracting FFCV files.
    """

    def __init__(self, config, split, moment2index, category2index, domain2index, holdout_domain='iiith', ood=False):
        self.config = config
        self.split = split
        self.moment2index = moment2index
        self.category2index = category2index
        self.domain2index = domain2index
        self.holdout_domain = holdout_domain
        self.ood = ood
        self.read_csv()

    def read_csv(self):
        print('*** read csv...')
        split_idx = self.config.dataset_splits.index(self.split)
        csv_name = self.config.dataset_csvs[split_idx]
        df = pd.read_csv(os.path.join(self.config.repo_root, 'annotations', csv_name),
                         sep="::", engine="python", 
                         header=None, 
                         names=["domain", "featpath", "category"])
        df = pd.concat([df, df['featpath'].apply(parse_featpath)], axis=1)
        
        if self.ood:
            df = df[df['domain'] == self.holdout_domain]
        else:
            df = df[df['domain'] != self.holdout_domain]

        self.items = df

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        data = self.items.iloc[index]
        uid = data['uid']
        startframe = data['start_frame']
        endframe = data['end_frame']
        domain = data['domain']
        moment = data['moment']
        category = data['category']

        domain_idx   = int(self.domain2index[domain])
        moment_idx   = int(self.moment2index[moment])
        category_idx = int(self.category2index[category])
        label = np.array([domain_idx, moment_idx, category_idx], dtype=np.int32)

        feat_path = os.path.join(self.config.feat_root, self.split, data['featpath'])
        with open(feat_path, 'rb') as f:
            feat = np.load(f)

        # sample 3 evenly-spaced temporal indices from the 15-stride window
        selection_idxs = np.clip([0, 7, 14], 0, feat.shape[0] - 1)
        feat_selection = feat[selection_idxs]

        return (uid, int(startframe), int(endframe)), feat_selection, label

