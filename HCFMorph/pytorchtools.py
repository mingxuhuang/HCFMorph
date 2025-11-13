import os.path

import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, name,savepath,patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print,val_loss_min = np.Inf):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.savepath = savepath
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.equal = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = val_loss_min
        self.delta = delta
        self.path = path
        self.name = name
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            # plot(allval,score)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} {self.equal} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        elif score == self.best_score + self.delta:
            self.equal += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} {self.equal} out of {self.patience}')
            if self.equal >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            # plot(allval, score)
            self.counter = 0
            self.equal = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(self.savepath,self.name+self.path))
        self.val_loss_min = val_loss


# def plot(all,loss):
#     head = ['sen', 'spe', 'pre', 'dice', 'ACC', 'b_dice', 'th', 'methods']
#     all = (np.array(all).T).tolist()
#     df = pd.DataFrame(all, columns=head, dtype=float)
#     for i in range(len(head)-3):
#         g = sns.boxplot(x="th", y=head[i], hue="methods",
#                         data=df, palette="Set3")
#         plt.suptitle('loss_'+str(loss), fontsize=20, color='red', backgroundcolor='yellow')
#         plt.show()