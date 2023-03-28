import numpy as np
import torch
import numpy as np
import pytorch_lightning as pl
from utils.word_vectorizer import WordVectorizer
from blip2.dataset.finetune_dataset import DATALoader, TextGenerateDATALoader

class FinetuneDataModule(pl.LightningDataModule):
    def __init__(self, args, dataname=None) -> None:
        super().__init__()
        self.args = args
        self.dataname = dataname
    
    def train_dataloader(self):
        train_loader = DATALoader(split='train', batch_size=self.args.batch_size, nodebug=self.args.nodebug)
        return train_loader
    
    def val_dataloader(self):
        val_loader = DATALoader(split='test', batch_size=self.args.eval_batch_size, nodebug=self.args.nodebug)
        return val_loader
    
    def test_dataloader(self):
        if self.dataname: # for generate text prompt
            test_loader = TextGenerateDATALoader(dataname=self.dataname, batch_size=self.arg.eval_batch_size, nodebug=self.args.nodebug)
        else:
            test_loader = DATALoader(split='test', batch_size=self.args.eval_batch_size, nodebug=self.args.nodebug)
        return test_loader