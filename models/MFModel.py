import torch.nn as nn
import torch
class MFModel(nn.Module):
    def __init__(self,user_list,item_list, hidden_dim) -> None:
        super(MFModel, self).__init__()
        self.user_list = user_list
        self.item_list = item_list
        self.hidden_dim = hidden_dim
        self.N=len(user_list)
        self.M=len(item_list)
        self.user_hiddens = nn.ParameterList([nn.Parameter(torch.rand(hidden_dim),requires_grad=False) for u in range(self.N)])
        self.item_hiddens = nn.ParameterList([nn.Parameter(torch.rand(hidden_dim),requires_grad=False) for i in range(self.M)])
        self.mode="user"
    def forward(self, user_ids:list, item_ids:list, mode:str="user"):
        if mode=="user":
            for u in user_ids:
                self.user_hiddens[u].requires_grad_(True)
            for i in item_ids:
                self.item_hiddens[i].requires_grad_(False)
        elif mode=="item":
            for u in user_ids:
                self.user_hiddens[u].requires_grad_(False)
            for i in item_ids:
                self.item_hiddens[i].requires_grad_(True)
        else:
            assert False
        out = []
        for u in user_ids:
            row=[]
            for i in item_ids:
                row.append(torch.dot(self.user_hiddens[u], self.item_hiddens[i]))
            out.append(row)
        return out
    def switch_tranining_mode(self, mode="user"):
        self.mode=mode
        if mode=="user":
            for embed in self.user_hiddens:
                embed.requires_grad_(True)
            for embed in self.item_hiddens:
                embed.requires_grad_(False)
        else:
            for embed in self.user_hiddens:
                embed.requires_grad_(False)
            for embed in self.item_hiddens:
                embed.requires_grad_(True)
    # def parameters(self):
    #     a=super().parameters()
    #     return filter(lambda x:x.requires_grad ,a)
    # def named_parameters(self):
    #     a=super().named_parameters()
    #     return filter(lambda x:x[1].requires_grad ,a)


        