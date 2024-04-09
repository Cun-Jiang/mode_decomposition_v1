import torch
import torch.nn as nn

# target=torch.tensor([1,1,1],dtype=torch.float64,requires_grad=True)
# output=torch.tensor([1,2,3],dtype=torch.float64,requires_grad=True)
# loss=nn.MSELoss()
# loss_value=loss(target,output)
# print(loss_value)
# a=torch.tensor([[1],[2],[3]],dtype=torch.float64)
# fhi=torch.tensor([[0],[1],[1]],dtype=torch.float64)
# output=a*torch.exp(fhi)
# print(a.shape)
# print(fhi.shape)
# print(output.shape)
# print(output)

# a=torch.rand(3,4)
# b=a[:,:2]
# c=a[:,2:]
# print(a)
# print(b)
# print(c)
# d=b*torch.exp(c)
# print(d)
# print(d.unsqueeze(2))

a=torch.rand(4,3)
b=torch.rand(4,3)
print(a)
print(b)
c=torch.add(a,b)
print(c)