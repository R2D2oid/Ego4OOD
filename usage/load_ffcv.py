from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import NDArrayDecoder
from ffcv.transforms import ToTensor
from ffcv.fields.bytes import BytesDecoder
from ffcv.fields.decoders import NDArrayDecoder, BytesDecoder
import json
from tqdm import tqdm
import argparse
import wandb
import os

class Learner:
    def __init__(self):
        self.parse_command_line()
        self.wb = wandb.init(config=self.args.config, mode=self.args.wandb_mode, project='DGB_Ego4OOD')
        self.config = self.wb.config

        self.init_data()

    def parse_command_line(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", "-c", default="ffcv_config.yaml", help="wandb config file.")
        parser.add_argument("--wandb_mode", "-wb", choices=["online", "offline", "disabled"], default="online",
                            help="disable wandb logging")
        self.args = parser.parse_args()

    def init_data(self):
        self.loaders = {}

        pipelines = {
            'info': [BytesDecoder()],
            'feat': [NDArrayDecoder(), ToTensor()],
            'label': [NDArrayDecoder(), ToTensor()]
        }

        for split_idx, split in enumerate(self.config.dataset_splits):
            ds_path = os.path.join(self.config.ffcv_path, self.config.dataset_ffcvs[split_idx])
            if split == "train":
                order = getattr(OrderOption, self.config.ffcv_order)
            else:
                order = OrderOption.SEQUENTIAL

            self.loaders[split] = Loader(
                ds_path,
                batch_size=self.config.batch_size,
                num_workers=self.config.n_workers,
                order=order,
                pipelines=pipelines
            )

    def read_data(self):
        for batch_idx, batch in tqdm(enumerate(self.loaders["train"])):
            info_raw = batch[0]          # uint8 array
            feat = batch[1]              # tensor [B, 3, 2304]
            label = batch[2]             # tensor [B, 3]

            # decode JSONField
            info = []
            for row in info_raw:
                s = bytes(row[row != 0]).decode('utf-8')
                info.append(json.loads(s))

            print("INFO:", info[0])
            print("FEAT:", feat.shape)
            print("LABEL:", label.shape)


# python load_ffcv.py --wandb_mode offline --config ffcv_config.yaml
def main():
    learner = Learner()
    learner.read_data()

if __name__ == "__main__":
    main()
