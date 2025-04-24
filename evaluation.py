import numpy as np
import torch
import torch.nn.functional as F

def calculate_mrr(ranked_items, true_items):
    """计算Mean Reciprocal Rank (MRR)"""
    for rank, item in enumerate(ranked_items, 1):
        if item in true_items:
            # print(f"{item} hit rank is {rank}")
            return 1.0 / rank
    return 0.0

def calculate_hr(ranked_items, true_items, k):
    """计算Hit Ratio (HR)"""
    hits = len(set(ranked_items) & set(true_items))
    return 1.0 if hits > 0 else 0.0

def softmax_scores(scores):
    """将得分转换为概率分布"""
    # 使用 PyTorch 的 Softmax 函数处理得分
    scores_tensor = torch.tensor(scores)
    probabilities = F.softmax(scores_tensor, dim=0)
    return probabilities.cpu().detach().numpy()  # 将结果转回 NumPy 数组


def evaluate_train(model, train_interaction_matrix, item_list,k=3, batch_size=64):
    print("test train")
        # 合并稀疏张量
    train_interaction_matrix = train_interaction_matrix.coalesce()

    hr_list = []
    mrr_list = []
    similarity_list = []
    
    # 获取测试集中的用户和物品索引
    user_indices = train_interaction_matrix.indices()[0, :]
    item_indices = train_interaction_matrix.indices()[1, :]
    users_to_evaluate = torch.unique(user_indices)
    
    # 批量评估用户
    for i in range(0, len(users_to_evaluate), batch_size):
        # 批量获取用户
        batch_users = users_to_evaluate[i:i + batch_size]
        
        # 批量生成所有用户的预测分数
        with torch.no_grad():
            out = model(batch_users.tolist(), item_list, mode="eval")
        
        for index in batch_users:
            # 提取当前用户的预测分数（点积结果）
            scores = out[batch_users.tolist().index(index)].cpu().detach().numpy().flatten()
            
            
            # 排序并推荐Top-K物品
            ranked_items = np.argsort(-scores)[:k]
            
            # 获取真实测试物品
            true_items = item_indices[user_indices == index].tolist()
            if not true_items:
                continue
            
            # 计算HR
            hr = calculate_hr(ranked_items, true_items, k)
            hr_list.append(hr)
            # 计算MRR
            mrr = calculate_mrr(ranked_items, true_items)
            mrr_list.append(mrr)
            
            # 计算真实物品的得分均值

            # 使用 Softmax 得分
            probability_scores = softmax_scores(scores)
            acc = [probability_scores[i] for i in true_items]
            similarity_list.append(np.mean(acc))  # 假设将所有物品的概率的均值作为相似度

    # 计算最终的HR、MRR、真实物品得分和相似度
    return {
        f"HR@{k}": np.mean(hr_list),
        f"MRR@{k}": np.mean(mrr_list),
        "Avg Similarity": np.mean(similarity_list)  # 概率的均值作为相似度
    }

def evaluate_recommendations(model, test_interaction_matrix, train_interaction_matrix, item_list,k=3, batch_size=64):
    # 合并稀疏张量
    print("test test")
    test_interaction_matrix = test_interaction_matrix.coalesce()
    train_interaction_matrix = train_interaction_matrix.coalesce()

    hr_list = []
    mrr_list = []
    similarity_list = []
    
    # 获取测试集中的用户和物品索引
    user_indices = test_interaction_matrix.indices()[0, :]
    item_indices = test_interaction_matrix.indices()[1, :]
    users_to_evaluate = torch.unique(user_indices)
    # ids=[]
    # print(users_to_evaluate)
    # for i in users_to_evaluate:
    #     ids.append(complete_user_list[i.item()])
    # print(ids)
    # cold_start_users = 0
    # for user in users_to_evaluate:
    #     train_items = train_interaction_matrix[user].to_dense().sum()
    #     if train_items == 0:
    #         cold_start_users += 1
    # print(f"冷启动用户比例: {cold_start_users / len(users_to_evaluate):.2%}")
    # model.eval()  # Ensure model is in evaluation mode
    
    # 批量评估用户
    for i in range(0, len(users_to_evaluate), batch_size):
        # 批量获取用户
        batch_users = users_to_evaluate[i:i + batch_size]
        
        # 批量生成所有用户的预测分数
        with torch.no_grad():
            out = model(batch_users.tolist(), item_list, mode="eval")
        
        for index in batch_users:
            # 提取当前用户的预测分数（点积结果）
            scores = out[batch_users.tolist().index(index)].cpu().detach().numpy().flatten()
            
            # 排除训练集中的物品
            user_train_mask = train_interaction_matrix[index].to_dense().bool().cpu().numpy()
            num_train_items = user_train_mask.sum()
            # 如果所有训练交互为 False，则输出警告
            if num_train_items == 0:
                print(f"WARNING: User {index} has ZERO training interactions!")
            valid_scores = np.where(user_train_mask, -np.inf, scores)  # 将已交互的物品分数设为负无穷
            
            # 排序并推荐Top-K物品
            ranked_items = np.argsort(-valid_scores)[:k]
            
            # 获取真实测试物品
            true_items = item_indices[user_indices == index].tolist()
            if not true_items:
                continue
            
            # 计算HR
            hr = calculate_hr(ranked_items, true_items, k)
            hr_list.append(hr)
            if(len(set(ranked_items) & set(true_items))>0):
                print(f"user:{index} recommend {ranked_items} true is {true_items}")
                print(f"user:{index} hit {set(ranked_items) & set(true_items)}")
            # 计算MRR
            mrr = calculate_mrr(ranked_items, true_items)
            mrr_list.append(mrr)
            
            # 计算真实物品的得分均值

            # 使用 Softmax 得分
            probability_scores = softmax_scores(scores)
            acc = [probability_scores[i] for i in true_items]
            similarity_list.append(np.mean(acc))  # 假设将所有物品的概率的均值作为相似度

    # 计算最终的HR、MRR、真实物品得分和相似度
    return {
        f"HR@{k}": np.mean(hr_list),
        f"MRR@{k}": np.mean(mrr_list),
        "Avg Similarity": np.mean(similarity_list)  # 概率的均值作为相似度
    }
