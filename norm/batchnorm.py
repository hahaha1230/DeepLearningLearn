import torch
from torch import nn

if __name__ == '__main__':
    m = nn.BatchNorm1d(5, affine=False, momentum=0.1)
    tensor = torch.FloatTensor([i for i in range(20)]).reshape(4, 5)
    print(tensor)
    output = m(tensor)
    print(output)
    print(m.running_mean)
    print(m.running_var)
    print(output.runing_var)