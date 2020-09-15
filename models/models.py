import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from scipy.signal import max_len_seq
from .binarize_modules import *

def seq_matrix(period,rotate):
    M_seq = np.array(max_len_seq(period)[0])
    #M_seq = np.concatenate([M_seq,M_seq[:1]])
    M_Matrix=np.copy(M_seq)
    for i in range(rotate-1):
        M_seq=np.roll(M_seq,1)
        M_Matrix=np.vstack((M_Matrix,M_seq))
    M_Mat=torch.from_numpy(M_Matrix)
    return M_Mat


class MLPBinaryConnect(nn.Module):
    """Multi-Layer Perceptron used for MNIST. No convolution layers.
       This model is the MLP architecture used in paper "An empirical study of Binary NN optimization".
    """
    def __init__(self, in_features, out_features, num_units=2048, momentum=0.15, eps=1e-4,drop_prob=0,batch_affine=True):
        super(MLPBinaryConnect, self).__init__()
        self.in_features = in_features
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.dropout2 = nn.Dropout(p=drop_prob)


        self.dropout3 = nn.Dropout(p=drop_prob)

        self.dropout4 = nn.Dropout(p=drop_prob)


        self.fc1 = BinaryLinear(in_features, num_units, bias=False)
        self.bn1 = nn.BatchNorm1d(num_units, eps=eps, momentum=momentum,affine=batch_affine)

        self.fc2 = BinaryLinear(num_units, num_units, bias=False)
        self.bn2 = nn.BatchNorm1d(num_units, eps=eps, momentum=momentum,affine=batch_affine)

        self.fc3 = BinaryLinear(num_units, num_units, bias=False)
        self.bn3 = nn.BatchNorm1d(num_units, eps=eps, momentum=momentum,affine=batch_affine)
        

        self.fc4 = BinaryLinear(num_units, out_features, bias=False)
        self.bn4 = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum,affine=batch_affine)


    def forward(self, x):
        x = x.view(-1, self.in_features)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = F.relu(self.bn1(x))
        x = self.dropout2(x)


        x = self.fc2(x)
        x = F.relu(self.bn2(x))
        x = self.dropout3(x)


        x = self.fc3(x)
        x = F.relu(self.bn3(x))
        x = self.dropout4(x)

        x = self.fc4(x)
        x = self.bn4(x)

        return x  # if used NLL loss, the output should be changed to F.log_softmax(x,dim=1),
    
    
class MLPBinaryConnect_M1(nn.Module):
    """Multi-Layer Perceptron used for MNIST. No convolution layers.
       This model is the MLP architecture used in paper "An empirical study of Binary NN optimization".
    """
    def __init__(self, in_features, out_features, num_units=2048, momentum=0.15, eps=1e-4,drop_prob=0,batch_affine=True):
        super(MLPBinaryConnect_M1, self).__init__()
        self.in_features = in_features
        self.seq_data = seq_matrix(11,out_features) #test line
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.dropout2 = nn.Dropout(p=drop_prob)


        self.dropout3 = nn.Dropout(p=drop_prob)

        self.dropout4 = nn.Dropout(p=drop_prob)


        self.fc1 = BinaryLinear(in_features, num_units, bias=False)
        self.bn1 = nn.BatchNorm1d(num_units, eps=eps, momentum=momentum,affine=batch_affine)

        self.fc2 = BinaryLinear(num_units, num_units, bias=False)
        self.bn2 = nn.BatchNorm1d(num_units, eps=eps, momentum=momentum,affine=batch_affine)

        self.fc3 = BinaryLinear(num_units, num_units, bias=False)
        self.bn3 = nn.BatchNorm1d(num_units, eps=eps, momentum=momentum,affine=batch_affine)
        
        self.mseq = BinaryLinear(num_units, out_features,bias=False)
        self.mseq.requires_grad_(False)
        self.mseq.weight.copy_(self.seq_data)
        
        

        self.fc4 = BinaryLinear(num_units, out_features, bias=False)
        self.bn4 = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum,affine=batch_affine)


    def forward(self, x):
        x = x.view(-1, self.in_features)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = F.relu(self.bn1(x))
        x = self.dropout2(x)


        x = self.fc2(x)
        x = F.relu(self.bn2(x))
        x = self.dropout3(x)


        x = self.fc3(x)
        x = F.relu(self.bn3(x))
        x = self.dropout4(x)

        x = self.mseq(x)
        x = self.bn4(x)

        return x  # if used NLL loss, the output should be changed to F.log_softmax(x,dim=1),

    
class MLPBinaryConnect_M2(nn.Module):
    """Multi-Layer Perceptron used for MNIST. No convolution layers.
       This model is the MLP architecture used in paper "An empirical study of Binary NN optimization".
    """
    def __init__(self, in_features, out_features, num_units=2048, momentum=0.5, eps=1e-4,drop_prob=0,batch_affine=True):
        super(MLPBinaryConnect_M2, self).__init__()
        self.in_features = in_features
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.dropout2 = nn.Dropout(p=drop_prob)
        
        self.seq_data = seq_matrix(11,num_units)
        
        self.dropout3 = nn.Dropout(p=drop_prob)

        self.dropout4 = nn.Dropout(p=drop_prob)


        self.fc1 = BinaryLinear(in_features, num_units, bias=False)
        self.bn1 = nn.BatchNorm1d(num_units, eps=eps, momentum=momentum,affine=batch_affine)

        self.fc2 = BinaryLinear(num_units, num_units, bias=False)
        self.bn2 = nn.BatchNorm1d(num_units, eps=eps, momentum=momentum,affine=batch_affine)

        self.fc3 = BinaryLinear(num_units, num_units, bias=False)
        self.bn3 = nn.BatchNorm1d(num_units, eps=eps, momentum=momentum,affine=batch_affine)
        
        self.mseq = nn.Linear(num_units, num_units,bias=False)
        self.mseq.requires_grad_(False)
        self.mseq.weight.copy_(self.seq_data)

        self.fc4 = BinaryLinear(num_units, out_features, bias=False)
        self.bn4 = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum,affine=batch_affine)


    def forward(self, x):
        x = x.view(-1, self.in_features)
        x = self.dropout1(x)
        x = self.mseq(x)
        x = F.relu(self.bn1(x))
        x = self.dropout2(x)


        x = self.fc2(x)
        x = F.relu(self.bn2(x))
        x = self.dropout3(x)


        x = self.fc3(x)
        x = F.relu(self.bn3(x))
        x = self.dropout4(x)

        

        x = self.fc4(x)
        x = self.bn4(x)

        return x  # if used NLL loss, the output should be changed to F.log_softmax(x,dim=1),
    
class MLPBinaryConnect_M3(nn.Module):
    """Multi-Layer Perceptron used for MNIST. No convolution layers.
       This model is the MLP architecture used in paper "An empirical study of Binary NN optimization".
    """
    def __init__(self, in_features, out_features, num_units=2048, momentum=0.15, eps=1e-4,drop_prob=0,batch_affine=True):
        super(MLPBinaryConnect_M1, self).__init__()
        self.in_features = in_features
        self.seq_data = torch. ones(num_units, num_units, requires_grad=False) #test line
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.dropout2 = nn.Dropout(p=drop_prob)


        self.dropout3 = nn.Dropout(p=drop_prob)

        self.dropout4 = nn.Dropout(p=drop_prob)


        self.fc1 = BinaryLinear(in_features, num_units, bias=False)
        self.bn1 = nn.BatchNorm1d(num_units, eps=eps, momentum=momentum,affine=batch_affine)

        self.fc2 = BinaryLinear(num_units, num_units, bias=False)
        self.bn2 = nn.BatchNorm1d(num_units, eps=eps, momentum=momentum,affine=batch_affine)

        self.fc3 = BinaryLinear(num_units, num_units, bias=False)
        self.bn3 = nn.BatchNorm1d(num_units, eps=eps, momentum=momentum,affine=batch_affine)
        
        self.mseq = nn.Linear(num_units, num_units,bias=False)
        self.mseq.requires_grad_(False)
        self.mseq.weight.copy_(self.seq_data)

        self.fc4 = BinaryLinear(num_units, out_features, bias=False)
        self.bn4 = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum,affine=batch_affine)


    def forward(self, x):
        x = x.view(-1, self.in_features)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = F.relu(self.bn1(x))
        x = self.dropout2(x)


        x = self.fc2(x)
        x = F.relu(self.bn2(x))
        x = self.dropout3(x)


        x = self.fc3(x)
        x = F.relu(self.bn3(x))
        x = self.dropout4(x)

        x = self.mseq(x)

        x = self.fc4(x)
        x = self.bn4(x)

        return x  # if used NLL loss, the output should be changed to F.log_softmax(x,dim=1),