import numpy as np
import torch
import numpy as np
import pytorch_lightning as pl
from utils.word_vectorizer import WordVectorizer
from dataset.dataset_TM_train import DATALoader as train_dl
from dataset.dataset_TM_eval import DATALoader as eval_dl

class Humanml3dDataModule(pl.LightningDataModule):
    def __init__(self, args) -> None:
        super().__init__()
    
        self.args = args
        self.w_vectorizer = WordVectorizer('./glove', 'our_vab')
    
    def train_dataloader(self):
        if self.args.nodebug:
            train_loader = train_dl(self.args.dataname, self.args.batch_size, self.args.nb_code, self.args.vq_name, unit_length=2**self.args.down_t, nodebug=self.args.nodebug)
        else:
            train_loader = train_dl(self.args.dataname, self.args.batch_size, self.args.nb_code, self.args.vq_name, unit_length=2**self.args.down_t, num_workers=0, nodebug=self.args.nodebug)
        return train_loader
    
    def val_dataloader(self):
        # overrides batch_size and num_workers

        return eval_dl(
            self.args.dataname, False, 32, self.w_vectorizer, self.args.nodebug
        )
    
    def test_dataloader(self):
        return eval_dl(
            self.args.dataname, True, 32, self.w_vectorizer, self.args.nodebug
        )