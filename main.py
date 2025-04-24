import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
import json
import pandas as pd
from torch.autograd import Variable
import torch
from models.MFModel import MFModel, MFModel_light
import random
import torch.nn as nn
import pickle
from evaluation import evaluate_recommendations,evaluate_train
import os
# from utility.parser import parse_args
# args = parse_args()
path="data/NETFLIX/raw"
import time
from utils import load_model

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
        # interaction_matrix = sp.dok_matrix((len(user_list), len(item_list)), dtype=np.float32)
        index_row = []
        index_col = []
        value = []
        # print(interations.items())
        for uid, items in interations.items():
            # print(uid,end=" ")
            u = user_list.index(str(uid))
            for item_id in items:
                i = item_list.index(str(item_id))
                index_row.append(u)
                index_col.append(i)
                value.append(1)
        
        indices= torch.tensor([index_row, index_col], dtype=torch.long)  # 行索引和列索引
        interaction_matrix=torch.sparse_coo_tensor(indices, value, (len(user_list), len(item_list)))
        
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
    test_interaction_matrix= get_interaction_matrix(complete_item_list,complete_user_list,test_interaction_data)
    return interaction_matrix,test_interaction_matrix, complete_item_list,complete_user_list
    
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

    for name_and_param, grad in zip(filter(lambda x:x[1].requires_grad, model.named_parameters()), filter(lambda x:x is not None,grads)):
        name, param = name_and_param
        if grad is None:
            param.data.sub_(0)
        else:
            # print(name, param, grad)
            param.data.sub_(learning_rate * grad)
    global time8
    time8=time.time()
def sample_a_batch_of_ids(batch_size:int,  id_pool:list):
    ids = id_pool.copy()
    if batch_size<len(id_pool):
        ids=random.sample(id_pool,batch_size)
    for id in ids:
        id_pool.remove(id)
    return ids

def update_model_light(model:MFModel_light, loss, learning_rate):
    global time6
    time6=time.time()
    paras = model.get_para()
    grads = torch.autograd.grad(
        outputs=loss, inputs= paras
    )
    
    global time7
    time7=time.time()
    # print(grads[0])
    for i in range(len(paras)):
        param=paras[i]
        # print(grads[0][i])
        param.data.sub_(learning_rate * grads[0][i])
    model.set_hidden()
    global time8
    time8=time.time()
time3=0
time4=0
time5=0


def calculate_loss(model:nn.Module, input_user_ids:list, input_items_ids:list, mode:str, interaction_matrix):
    global time3
    time3=time.time()
    out=model(input_user_ids, input_items_ids, mode)
    global time4
    time4=time.time()

    loss=None
    for u_idx, u in  enumerate(input_user_ids,0):
        for i_idx,i in enumerate(input_items_ids,0):
            if loss is None:
                loss=(out[u_idx][i_idx] - interaction_matrix[u,i])**2
                # print(type(loss), loss.device)
                # assert False
            else:
                loss+=(out[u_idx][i_idx] - interaction_matrix[u,i])**2

    global time5
    time5=time.time()
    return loss

def calculate_loss_light(model:MFModel_light, input_user_ids:list, input_items_ids:list, mode:str, interaction_matrix):
    global time3
    time3=time.time()
    out=model(input_user_ids, input_items_ids, mode)
    global time4
    
    time4=time.time()


    interaction_matrix_shard = interaction_matrix.index_select(dim=0 if mode == "user" else 1, index=torch.tensor(input_user_ids if mode == "user" else input_items_ids)).to_dense().to(device)
    # print(interaction_matrix_shard.shape, out.shape)
    # a=interaction_matrix_shard-out
    # print(a.shape)
    # interaction_matrix_shard = torch.zeros((len(input_user_ids), len(input_items_ids))).to(device)
    # for u_idx,u in enumerate(input_user_ids,0):
    #     for i_idx,i in  enumerate(input_items_ids,0):
    #         interaction_matrix_shard[u_idx][i_idx]= torch.tensor(interaction_matrix[u,i])

    global time5
    time5=time.time()
    loss=None
    loss = torch.norm(interaction_matrix_shard - out)+  0* torch.norm(model.get_para())
    return loss


def load_pretrained_model(model, model_path):
    if os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}...")
        with open(model_path, "rb") as f:
            checkpoint = pickle.load(f)
            model.user_hiddens = checkpoint["user"]
            model.item_hiddens = checkpoint["item"]
        print("Pre-trained model loaded successfully.")
    else:
        print(f"No pre-trained model found at {model_path}, starting training from scratch.")

# training and model update functions...

if __name__ == "__main__":
    hidden_dim = 8
    total_epoch_number = 1000
    batch_size = 256
    lr = 10
    os.chdir("/home/linfengyang/mmfedrec")
    model_path = "trained_models/onlyinteract_epoch1000-lr10-hidden8.pkl"

    interaction_matrix, test_interaction_matrix, complete_item_list, complete_user_list = ensure_consistency_of_items_across_datasets()
    model = MFModel_light(complete_user_list, complete_item_list, hidden_dim).to(device)
    N = len(complete_user_list)
    M = len(complete_item_list)

    # Load the pre-trained model (if exists)
    load_pretrained_model(model, model_path)
    # demo_user = 0
    # demo_item = 0
    # Training process...
    for epoch in range(total_epoch_number):
        # Reset id pool
        # user_id_pool = [u for u in range(N)]
        # item_id_pool = [i for i in range(M)]
        user_id_pool = torch.unique(interaction_matrix.coalesce().indices()[0, :]).numpy().tolist()

        item_id_pool = [i for i in range(M)]
        time1 = time.time()
       

        # 每训练10轮后打印
        # if epoch % 10 == 0:
        #     print(f"Epoch {epoch} 用户隐向量:", model.user_hiddens[demo_user][:5])
        #     print(f"Epoch {epoch} 物品隐向量:", model.item_hiddens[demo_item][:5])
        while len(user_id_pool) > 0:
            input_user_ids = sample_a_batch_of_ids(batch_size, user_id_pool)
            input_items_ids = item_id_pool

            loss = calculate_loss_light(model, input_user_ids, input_items_ids, "user", interaction_matrix)
            update_model_light(model, loss, lr / M)

            if len(user_id_pool) == 0:
                log_info = "Epoch: {}-{} \t{}/{} \tLoss: {:.6f}".format(epoch, "user", N - len(user_id_pool), N, loss.item())
                print(log_info)

        # Item-wise updates
        # user_id_pool = [u for u in range(N)]
        # item_id_pool = [i for i in range(M)]
        user_id_pool = [u for u in range(N)]
        item_id_pool = torch.unique(interaction_matrix.coalesce().indices()[1, :]).numpy().tolist()
        while len(item_id_pool) > 0:
            input_user_ids = user_id_pool
            input_items_ids = sample_a_batch_of_ids(batch_size, item_id_pool)
            loss = calculate_loss_light(model, input_user_ids, input_items_ids, "item", interaction_matrix)
            update_model_light(model, loss, lr / N)
            if len(item_id_pool) == 0:
                log_info = "Epoch: {}-{} \t{}/{} \tLoss: {:.6f}".format(epoch, "item", M - len(item_id_pool), M, loss.item())
                print(log_info)

        # Evaluate on test data
        # print(complete_user_list)
        metric = evaluate_recommendations(model, test_interaction_matrix, interaction_matrix,range(M),k=10, batch_size=64)
        print(f"test result:{metric}")
        metric = evaluate_train(model, interaction_matrix,range(M),k=10, batch_size=64)
        print(f"train result:{metric}")

    # Save the model
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    with open(model_path, "wb") as f:
        pickle.dump({"user": model.user_hiddens, "item": model.item_hiddens}, f)