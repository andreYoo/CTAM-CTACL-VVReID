import torch
import numpy as np

class Dictionary():
    def __init__(self, num_features, num_classes, alpha=0.01):
        super(Dictionary, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.alpha = alpha
        self.mem = torch.zeros(num_classes, num_features,dtype=torch.double)
        self.tpid_memory = np.zeros([num_classes],dtype=np.uint32)

    def store(self,inputs,target):
        self.mem[target]  = inputs.to('cpu')

    def update(self, inputs, targets):
        for x, y in zip(inputs, targets):
            self.mem[y] = self.mem[y] + x
            self.mem[y] /= self.mem[y].norm()

    def output(self, inputs):
        outputs = inputs.mm(self.mem.t())
        return outputs


