import torch.utils.data as data
import torch

import numpy as np
import unicodedata

class TextSet(data.Dataset):

    def __init__(self, files, seqlen=100, chars=None):
        self.text = ''
        self.chars = set()
        self.seqlen = seqlen
        for x in files:
            with open(x, 'r') as fp:
                t = unicodedata.normalize('NFD', fp.read())
                self.chars.update(t)
                self.text += t
        if not chars:
            self.chars = {c: i for i, c in enumerate(self.chars)}
        else:
            self.chars = chars
        self.oh_dim = len(self.chars)

    def __len__(self):
        return len(self.text)//self.seqlen

    def __getitem__(self, idx):
        """
        Returns a random one-hot encoded sequence of length seqlen.
        """
        start = np.random.randint(len(self.text) - self.seqlen - 1)
        labels = torch.tensor([self.chars[c] for c in self.text[start:start+self.seqlen]])
        target = torch.tensor([self.chars[c] for c in self.text[start+1:start+self.seqlen+1]])
        return torch.eye(self.oh_dim)[labels.long()].t().float(), torch.eye(self.oh_dim)[target.long()].t().float()
