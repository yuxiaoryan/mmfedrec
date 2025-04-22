import torch.nn as nn
import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


class MFModel_light(nn.Module):
    def __init__(self,user_list,item_list, hidden_dim) -> None:
        super(MFModel_light, self).__init__()
        self.user_list = user_list
        self.item_list = item_list
        self.hidden_dim = hidden_dim
        self.N=len(user_list)
        self.M=len(item_list)
        self.user_hiddens = [np.random.random(hidden_dim) for u in range(self.N)]
        self.item_hiddens = [np.random.random(hidden_dim) for i in range(self.M)]

        # self.user_hiddens = nn.ParameterList([nn.Parameter(torch.rand(hidden_dim),requires_grad=False) for u in range(self.N)])
        # self.item_hiddens = nn.ParameterList([nn.Parameter(torch.rand(hidden_dim),requires_grad=False) for i in range(self.M)])
        self.mode="user"
    def forward(self, user_ids:list, item_ids:list, mode:str="user"):
        self.curr_user_ids = user_ids
        self.curr_item_ids = item_ids
        self.mode = mode
        if mode=="user":
            self.user_hidden_ = nn.Parameter(torch.tensor(np.array([self.user_hiddens[u] for u in user_ids]))).to(device)
            self.item_hidden_ = torch.tensor(np.array([self.item_hiddens[i] for i in item_ids])).to(device)
        elif mode=="item":
            self.user_hidden_ = torch.tensor(np.array([self.user_hiddens[u] for u in user_ids])).to(device)
            self.item_hidden_ = nn.Parameter(torch.tensor(np.array([self.item_hiddens[i] for i in item_ids]))).to(device)
        else:
            assert False

        out = torch.mm(self.user_hidden_, self.item_hidden_.T)
        return out
    def get_para(self):
        if self.mode=="user":
            return self.user_hidden_
        elif self.mode=="item":
            return self.item_hidden_
        else:
            assert False
    def set_hidden(self):
        if self.mode=="user":
            for u_idx, u in enumerate(self.curr_user_ids):
                self.user_hiddens[u] =  np.array(self.user_hidden_[u_idx].detach().cpu())
        elif self.mode=="item":
            for i_idx, i in enumerate(self.curr_item_ids):
                self.item_hiddens[i] =  np.array(self.item_hidden_[i_idx].detach().cpu())

    def load_hiddens(self, u_hiddens, i_hiddens):
        del self.user_hiddens
        del self.item_hiddens
        self.user_hiddens = u_hiddens
        self.item_hiddens = i_hiddens