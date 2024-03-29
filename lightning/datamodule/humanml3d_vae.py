import numpy as np
import torch
import numpy as np
import pytorch_lightning as pl
from utils.word_vectorizer import WordVectorizer
from dataset.dataset_VQ import DATALoader as train_dl
from dataset.dataset_VQ import withH36mDATALoader as h36m_train_dl
from dataset.dataset_TM_eval import DATALoader as eval_dl

class Humanml3dVaeDataModule(pl.LightningDataModule):
    def __init__(self, args) -> None:
        super().__init__()
    
        self.args = args
        self.w_vectorizer = WordVectorizer('./glove', 'our_vab')
    
    def train_dataloader(self):
        dataloader_fn = h36m_train_dl if self.args.h36m else train_dl
        return dataloader_fn(
            self.args.dataname, self.args.batch_size, window_size=self.args.window_size, unit_length=2**self.args.down_t, nodebug=self.args.nodebug
        )
    
    def val_dataloader(self):
        return eval_dl(
            self.args.dataname, True, 32, self.w_vectorizer, nodebug=self.args.nodebug
        )
    
    def test_dataloader(self):
        return eval_dl(
            self.args.dataname, True, 32, self.w_vectorizer, nodebug=self.args.nodebug
        )