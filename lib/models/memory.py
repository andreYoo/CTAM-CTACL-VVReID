import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
import numpy as np
device_0 = torch.device('cuda:0')
device_1 = torch.device('cuda:1')

class MemoryLayer(Function):
    def __init__(self, memory):
        super(MemoryLayer, self).__init__()
        self.memory = memory
        self.global_norm = nn.BatchNorm1d(2048)

    def forward(self, inputs, targets):
        self.save_for_backward(inputs, targets)
        outputs = inputs.mm(self.memory.t())
        return outputs

    def backward(self, grad_outputs):
        inputs, targets = self.saved_tensors
        grad_inputs = None
        if self.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(self.memory)
        for x, y in zip(inputs, targets):
            self.memory[y] = 0.5*self.memory[y] + 0.5*x
            self.memory[y] /= self.memory[y].norm()
        return grad_inputs, None

class Memory(nn.Module):
    def __init__(self, num_features, num_classes,num_cam, alpha=0.01):
        super(Memory, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_cam = num_cam
        self.alpha = alpha
        self.global_norm = nn.BatchNorm1d(num_features)
        self.mem = nn.Parameter(torch.zeros(num_classes, num_features),requires_grad=False)
        self.mem_cam = nn.Parameter(torch.zeros(num_cam, num_features),requires_grad=False)
        self.mem_TID = torch.empty((num_classes), dtype=torch.long, device = device_1)
        self.mem_CID = torch.empty((num_classes), dtype=torch.long, device = device_1)

    def store(self,inputs,camid,tid,target):
        self.mem_TID[target] =  tid.to(device_1)
        self.mem_CID[target] =  camid.to(device_1)
        self.mem[target]  = inputs.to(device_1)

    def set_cam_memory(self):
        valid_cam_num = 0
        valid_cam_id = []
        for i in range(self.num_cam):
            tmp_set = self.mem[self.mem_CID==i]
            if tmp_set.size(0)==0:
                continue
            else:
                valid_cam_id.append(i)
                valid_cam_num +=1
                self.mem_cam[i] = torch.mean(tmp_set,0)
                self.mem_cam[i] /= self.mem_cam[i].norm()
                self.mem_cam[i] = self.mem_cam[i].to(device_0)
        return valid_cam_num, valid_cam_id

    def get_cam_likelihood(self,inputs):
        inputs = inputs.to(devide_1)
        return inputs.mm(self.mem_cam.t())
        

    def get_cam_mem(self,camid):
        return self.mem[self.mem_CID==camid],self.TID[self.mem_CID==camid]

    def forward(self, inputs,targets, epoch=None):
        #originally, no device assignment
        inputs = inputs.to(device_1)
        targets = targets.to(device_1)
        logits = MemoryLayer(self.mem)(inputs, targets)
        return logits
