import tqdm
import torch
from torch import nn
from torch import optim

from models import TKBCModel
from regularizers import Regularizer
from datasets import TemporalDataset

class TKBCOptimizer(object):
    def __init__(
            self, model: TKBCModel,
            emb_regularizer: Regularizer, temporal_regularizer: Regularizer,
            optimizer: optim.Optimizer, batch_size: int = 256,
            verbose: bool = True,use_freq_loss=False, freq_beta=1e-4
    ):
        self.model = model
        self.emb_regularizer = emb_regularizer
        self.temporal_regularizer = temporal_regularizer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose
        self.use_freq_loss = use_freq_loss
        self.freq_beta = freq_beta



    def epoch(self, examples: torch.LongTensor, epoch: int):
        actual_examples = examples[torch.randperm(examples.shape[0]), :]
        loss = nn.CrossEntropyLoss(reduction='mean')
        with tqdm.tqdm(total=examples.shape[0], unit='ex', disable=not self.verbose) as bar:
            bar.set_description(f'train loss')
            b_begin = 0
            while b_begin < examples.shape[0]:
                input_batch = actual_examples[
                    b_begin:b_begin + self.batch_size
                ].cuda()

                predictions, factors, time,low_freq,high_freq= self.model.forward(input_batch)
                truth = input_batch[:, 2]

                l_fit = loss(predictions, truth)
                l_reg = self.emb_regularizer.forward(factors)
                l_time = torch.zeros_like(l_reg)
                if time is not None:
                    l_time = self.temporal_regularizer.forward(time)
                 # ===== freq loss（完全独立）=====
                l_freq = 0.0
                if self.use_freq_loss and low_freq is not None:
                    l_freq = self.model.loss_freq(low_freq, high_freq)

                # ===== total loss =====
                if epoch < 50:
                    l = l_fit + l_reg + l_time
                else:
                    l = l_fit + l_reg + l_time + self.freq_beta * l_freq

                self.optimizer.zero_grad()
                l.backward()
                for param in self.model.parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any():
                            param.grad = torch.nan_to_num(param.grad)
                
                self.optimizer.step()
                b_begin += self.batch_size
                bar.update(input_batch.shape[0])
                print('loss={},reg={},cont={}'.format(l_fit.item(),l_reg.item(),l_time.item()))
                bar.set_postfix(
                fit=f'{l_fit.item():.3f}',
                reg=f'{l_reg.item():.3f}',
                time=f'{l_time.item():.3f}',
                freq=f'{float(l_freq):.3f}'
            )