import torch
print('hello')

# conda create -n CWP-assignment pandas numpy jupyter matplotlib
# source activate CWP-assignment

#conda deactivate

x = torch.randn(2, 3, 1)
print(x.shape)
y = torch.cat((x, x), dim=2)
print(y.shape)

print(y.permute(0,-1,-2).shape)