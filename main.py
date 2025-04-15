import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
import json
import pandas as pd
from torch.autograd import Variable
import torch
from models.MFModel import MFModel
import random
import torch.nn as nn
# from utility.parser import parse_args
# args = parse_args()
path="data/NETFLIX/raw"
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_interactions(dataset_type, generate_item_ids=False):
    train_file = path + f'/{dataset_type}.json'
    interaction_data = json.load(open(train_file))
    def get_items_in_interactions():
        item_list=[]
        user_list=[]
        if generate_item_ids:
            for uid, items in interaction_data.items():
                for item in items:
                    if item not in item_list:
                        item_list.append(item)
                    if uid not in user_list:
                        user_list.append(uid)
            with open(f"{path}/{dataset_type}_item_ids.txt", 'w') as f:
                f.write("\n".join(list(map(str,item_list))))
            with open(f"{path}/{dataset_type}_user_ids.txt", 'w') as f:
                f.write("\n".join(list(map(str,user_list))))
        else:
            with open(f"{path}/{dataset_type}_item_ids.txt", 'r') as f:
                item_list=[a.strip("\n") for a in f.readlines()]
            with open(f"{path}/{dataset_type}_user_ids.txt", 'r') as f:
                user_list=[a.strip("\n") for a in f.readlines()]
        return item_list,user_list
    
   
    item_list,user_list=get_items_in_interactions()
    return interaction_data,item_list,user_list


def ensure_consistency_of_items_across_datasets():
    def get_interaction_matrix(item_list, user_list, interations):
        interaction_matrix = sp.dok_matrix((len(user_list), len(item_list)), dtype=np.float32)
        # print(interations.items())
        for uid, items in interations.items():
            # print(uid,end=" ")
            u = user_list.index(str(uid))
            for item_id in items:
                i = item_list.index(str(item_id))
                interaction_matrix[u,i]+=1
        return interaction_matrix

    item_attribute_sheet=pd.read_csv(f"{path}/item_attribute_filter.csv",header=None)
    train_interaction_data, train_items, train_users = load_interactions("train", False)
    test_interaction_data, test_items, test_users = load_interactions("test",False)

    items_in_sheet= set(map(str,item_attribute_sheet[0]))
    train_test_items= set(map(str,train_items+test_items))

    #The items appear in the interaction matrices but not in the sheet should be appended
    for item in train_test_items:
        if item not in items_in_sheet:
            item_attribute_sheet.loc[len(item_attribute_sheet)] = [int(item), -1, ""]
    
    complete_item_list =  list(map(str,item_attribute_sheet[0]))
    complete_user_list = list(set(map(str, train_users+test_users)))
    complete_item_list.sort()
    complete_user_list.sort()
    interaction_matrix= get_interaction_matrix(complete_item_list,complete_user_list,train_interaction_data)
    return interaction_matrix, complete_item_list,complete_user_list
    
    # print(len(item_attribute_sheet))
    # print(len(items_in_sheet.union(train_test_items)), len(items_in_sheet), len(train_test_items), len(tmp))
time6=0
time7=0
time8=0
def update_model(model:nn.Module, loss, learning_rate):
    global time6
    time6=time.time()
    grads = torch.autograd.grad(
        outputs=loss, inputs=filter(lambda x: x.requires_grad, model.parameters()), allow_unused=True
    )
    global time7
    time7=time.time()

    for name_and_param, grad in zip( filter(lambda x:x[1].requires_grad, model.named_parameters()), filter(lambda x:x is not None,grads)):
        name, param = name_and_param
        if grad is None:
            param.data.sub_(0)
        else:
            print(name, param, grad)
            param.data.sub_(learning_rate * grad)
    global time8
    time8=time.time()
def sample_a_batch_of_ids(batch_size:int,  id_pool:list):
    ids=random.sample(id_pool,batch_size)
    for id in ids:
        id_pool.remove(id)
    return ids
time3=0
time4=0
time5=0
def calculate_loss(model:nn.Module, input_user_ids:list, input_items_ids:list, mode:str):
    global time3
    time3=time.time()
    out=model(input_user_ids, input_items_ids,mode)
    global time4
    time4=time.time()

    #高耗时
    loss=None
    for u_idx, u in  enumerate(input_user_ids,0):
        for i_idx,i in enumerate(input_items_ids,0):
            if loss is None:
                loss=(out[u_idx][i_idx] - interaction_matrix[u,i])**2
                print(type(loss), loss.device)
                assert False
            else:
                loss+=(out[u_idx][i_idx] - interaction_matrix[u,i])**2

    global time5
    time5=time.time()
    return loss

if __name__ == "__main__":
    hidden_dim = 8
    total_epoch_number = 100
    batch_size=16
    lr=0.0001

    interaction_matrix, complete_item_list, complete_user_list = ensure_consistency_of_items_across_datasets()
    model = MFModel(complete_user_list, complete_item_list, hidden_dim).to(device)
    N = len(complete_user_list)
    M = len(complete_item_list)

    #training process
   
    
    for epoch in range(total_epoch_number):
        #reset id pool
        user_id_pool=[u for u in range(N)]
        item_id_pool = [i for i in range(M)]
        time1=time.time()
        # model.switch_tranining_mode("user")
        while len(user_id_pool)>0:
            input_user_ids=sample_a_batch_of_ids(batch_size,user_id_pool)
            input_items_ids=item_id_pool
            print(input_user_ids)
            time2=time.time()
            loss=calculate_loss(model,input_user_ids, input_items_ids, "user")
            time6=time.time()
            update_model(model, loss, lr)
            
            log_info = "Epoch: {}-{} \t{}\{} \tLoss: {:.6f}".format(epoch,"user",N-len(user_id_pool) ,N, loss.item())
            print(log_info)
            print("a:{:.3f} b:{:.3f} c:{:.3f} d:{:.3f} e:{:.3f} f:{:.3f} g:{:.3f}".format(time2 -time1, time3-time2, time4-time3, time5-time4, time6-time5, time7-time6,time8-time7))
            assert False

        # model.switch_tranining_mode("item")
        while len(item_id_pool)>0:
            input_user_ids=user_id_pool
            input_items_ids=sample_a_batch_of_ids(batch_size,item_id_pool)
            loss=calculate_loss(model,input_user_ids, input_items_ids, "item")
            
            update_model(model, loss, lr)
            
            log_info = "Epoch: {}\t{} \t{}\{} \tLoss: {:.6f}".format(epoch,"item",M-len(item_id_pool) ,N, loss.item())
            print(log_info)
