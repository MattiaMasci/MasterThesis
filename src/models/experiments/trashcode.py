
import torch
import torch.nn as nn


lin0 = nn.Linear(100,5)
lin1 = nn.Linear(100,8)
lin2 = nn.Linear(100,20)

seq = nn.Sequential()

seq.append(lin0)
seq.append(lin1)

seq2 = nn.Sequential()
[seq2.append(mod) for mod in seq]
print(seq2.get_submodule("0"))
print(seq2.get_submodule("1"))
print(seq2==seq)
print(seq2.get_submodule("0")==seq.get_submodule("0"))
print(seq2.get_submodule("1")==seq.get_submodule("1"))

seq2.append(lin2)

print(seq)
print(seq2)