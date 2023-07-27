import sys
import math
import time
import copy
import random
import argparse

import numpy as np
import pandas as pd

from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import manifold
import pickle
import matplotlib.ticker as ticke

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.metrics import auc,roc_curve, precision_recall_curve, average_precision_score, roc_auc_score
from sklearn.metrics import confusion_matrix,classification_report,f1_score

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

def set_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_centroid", default=5, help="the number of clusters")
    parser.add_argument("--num_anomaly_classes", choices=[1,2,3,4,5,6,7], default = 3, help = "the type of labeled anomalies")
    parser.add_argument("--stage_1_epochs", default = 30, help="the number of stage one epochs")
    parser.add_argument("--stage_2_epochs", default = 30, help="the number of stage two epochs")
    parser.add_argument("--kmeans_batch", default = 256, help="kmeans_batch_size")
    parser.add_argument("--stage_1_batch", default = 256, help="stage_1_batch_size")
    parser.add_argument("--stage_2_batch", default = 128, help="stage_2_batch_size")
    parser.add_argument("--anomaly_batch", default = 32, help="anomaly_batch_size")
    parser.add_argument("--ood_batch", default = 32, help="ood_batch_size")
    parser.add_argument("--stage_1_lr", default = 0.0001, help="stage_1_learning_rate")
    parser.add_argument("--stage_2_lr", default = 0.00001, help="stage_2_learning_rate")

    parser.add_argument("--embedded_dimension", default = 64, help="the embedding dimension")
    parser.add_argument("--feature_dimension", default = 196, help="the input feature dimension")
    parser.add_argument("--filter", default = 0.05, help="the number of potential anomalies")
    
    parser.add_argument("--loss_oe", default = 0.1, help="the Coefficient of loss_oe")
    parser.add_argument("--loss_re", default = 1, help="the Coefficient of loss_re")
    
    args = parser.parse_known_args()[0]
    return args

args = set_parser()

#读取训练集数据
train_labelled_data = pd.read_csv('./data/Labelled_data_' + str(args.num_anomaly_classes) + '.csv', index_col = 0)
train_unlabelled_data = pd.read_csv('./data/Unlabelled_data.csv', index_col = 0)
val_data = pd.read_csv('./data/val_data.csv', index_col = 0)
test_data = pd.read_csv('./data/test_data.csv', index_col = 0)

# 已知异常的种类
labelled_set = list(set(train_labelled_data.attack_cat))
print(labelled_set)
# unlabeled data的数据情况
print("================================================")
Counter(train_unlabelled_data.attack_cat)

# 数据集划分标签
# 未知异常为-2、正常为-1、已知异常为5、6、7
train_unlabelled_data.loc[~(train_unlabelled_data.attack_cat.isin(labelled_set)) & (train_unlabelled_data.attack_cat != 'Normal'), 'attack_cat'] = -2
train_unlabelled_data.loc[train_unlabelled_data.attack_cat == 'Normal', 'attack_cat'] = -1

test_data.loc[~(test_data.attack_cat.isin(labelled_set)) & (test_data.attack_cat != 'Normal'), 'attack_cat'] = -2
test_data.loc[test_data.attack_cat == 'Normal', 'attack_cat'] = -1

val_data.loc[~(val_data.attack_cat.isin(labelled_set)) & (val_data.attack_cat != 'Normal'), 'attack_cat'] = -2
val_data.loc[val_data.attack_cat == 'Normal', 'attack_cat'] = -1

for i in range(len(labelled_set)):
    train_unlabelled_data.loc[train_unlabelled_data.attack_cat == labelled_set[i], 'attack_cat'] = args.num_centroid + i
    train_labelled_data.loc[train_labelled_data.attack_cat == labelled_set[i], 'attack_cat'] = args.num_centroid + i
    test_data.loc[test_data.attack_cat == labelled_set[i], 'attack_cat'] = args.num_centroid + i
    val_data.loc[val_data.attack_cat == labelled_set[i], 'attack_cat'] = args.num_centroid + i
    
    
# 数据预处理
train_labelled_data = train_labelled_data.fillna(0)
train_unlabelled_data = train_unlabelled_data.fillna(0)
test_data = test_data.fillna(0)
val_data = val_data.fillna(0)


# load data
X_labelled_anomaly = train_labelled_data.drop(['attack_cat','label'], axis=1).values
Y_labelled_anomaly = train_labelled_data.loc[:,['attack_cat']].values

X_unlabelled = train_unlabelled_data.drop(['attack_cat','label'], axis=1).values
Y_unlabelled = train_unlabelled_data.loc[:,['attack_cat']].values

X_test = test_data.drop(['attack_cat','label'], axis=1).values
Y_test = test_data.loc[:,['attack_cat']].values

X_val = val_data.drop(['attack_cat','label'], axis=1).values
Y_val = val_data.loc[:,['attack_cat']].values


scaler = MinMaxScaler()
scaler = scaler.fit(X_unlabelled)
X_unlabelled = scaler.transform(X_unlabelled)
X_labelled_anomaly = scaler.transform(X_labelled_anomaly)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)

# 查看数据集的情况
print("The Size of Labeled Data:", train_labelled_data.shape)
print("The Size of Unlabeled Data:", train_unlabelled_data.shape)
print("The Size of Test Data:", test_data.shape)
print("The Size of Val Data:", val_data.shape)
print("=================================================================================")
print("Training Labeled Data:",Counter(train_labelled_data.attack_cat))
print("Training Unlabeled Data:",Counter(train_unlabelled_data.attack_cat))
print("Test Data:",Counter(test_data.attack_cat))
print("Val Data:",Counter(val_data.attack_cat))

def shuffle(X, Y, S):
    random_index = np.random.permutation(X.shape[0])
    return X[random_index], Y[random_index], S[random_index]
    
def shuffle_u(X, Y):
    random_index = np.random.permutation(X.shape[0])
    return X[random_index], Y[random_index] 
    
def getBatch(X, Y, BATCH_SIZE):
    while True:
        X, Y = shuffle_u(X, Y)
        for i in range(int(len(X)/BATCH_SIZE)):
            yield X[i*BATCH_SIZE:(i+1)*BATCH_SIZE], Y[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
    
def getBatchWeigt(X, Y, W, BATCH_SIZE):
    while True:
        X, Y, W = shuffle(X, Y, W)
        for i in range(int(len(X)/BATCH_SIZE)):
            yield X[i*BATCH_SIZE:(i+1)*BATCH_SIZE], Y[i*BATCH_SIZE:(i+1)*BATCH_SIZE], W[i*BATCH_SIZE:(i+1)*BATCH_SIZE] 


class SoftCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logit, label, weight=None):
        assert logit.size() == label.size(), "logit.size() != label.size()"
        dim = logit.dim()
        # 选出预测概率最大的那一类的概率
        max_logit = logit.max(dim - 1, keepdim=True)[0]
        logit = logit - max_logit
        exp_logit = logit.exp()
        exp_sum = exp_logit.sum(dim - 1, keepdim=True)
        prob = exp_logit / exp_sum
        log_exp_sum = exp_sum.log()
        neg_log_prob = log_exp_sum - logit

        if weight is None:
            weighted_label = label
        else:
            if weight.size() != (logit.size(-1),):
                raise ValueError(
                    "since logit.size() = {}, weight.size() should be ({},), but got {}".format(
                        logit.size(),
                        logit.size(-1),
                        weight.size(),
                    )
                )
            size = [1] * label.dim()
            size[-1] = label.size(-1)
            weighted_label = label * weight.view(size)
        # ctx为context的缩写，自定义的forward和backward第一个参数必须是ctx,上下文管理器
        # save_for_backward能够保存forward()静态方法中的张量,从而可以在backward()静态方法中调用
        ctx.save_for_backward(weighted_label, prob)
        out = (neg_log_prob * weighted_label).sum(dim - 1)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        weighted_label, prob = ctx.saved_tensors
        old_size = weighted_label.size()
        # num_classes
        K = old_size[-1]
        # batch_size
        B = weighted_label.numel() // K

        grad_output = grad_output.view(B, 1)
        weighted_label = weighted_label.view(B, K)
        prob = prob.view(B, K)
        grad_input = grad_output * (prob * weighted_label.sum(1, True) - weighted_label)
        grad_input = grad_input.view(old_size)
        return grad_input, None, None
    
def soft_cross_entropy(logit, label, weight=None, reduce=None, reduction="mean"):
    if weight is not None and weight.requires_grad:
        raise RuntimeError("gradient for weight is not supported")
    losses = SoftCrossEntropyFunction.apply(logit, label, weight)
    reduction = {
        True: "mean",
        False: "none",
        None: reduction,
    }[reduce]
    if reduction == "mean":
        return losses.mean()
    elif reduction == "sum":
        return losses.sum()
    elif reduction == "none":
        return losses
    else:
        raise ValueError("invalid value for reduction: {}".format(reduction))
    
class AutoEncoder(nn.Module):

    def __init__(self, input_size,  num_features):
        super(AutoEncoder, self).__init__()
        self.input_size = input_size
        self.num_features = num_features
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 500),
            nn.ReLU(),
            nn.Linear(500, 256),
            nn.ReLU(),
            nn.Linear(256, num_features),
        )

        self.decoder = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, 500),
            nn.ReLU(),
            nn.Linear(500, input_size)
        )
        
        # -----model initialization----- #
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        # -----feature embedding----- #
        x = x.view(-1, self.input_size)
        x_e = self.encoder(x)
        x_de = self.decoder(x_e)
        x_de = x_de.view(-1, 1, self.input_size)
        return x_e, x_de
    

class Classifier(nn.Module):

    def __init__(self, input_size,  num_features, num_classes):
        super(Classifier, self).__init__()
        self.input_size = input_size
        self.num_features = num_features
        self.encoder = nn.Sequential(
#             nn.Dropout(p=0.1),
            nn.Linear(input_size, 256),
            nn.ReLU(),
            # nn.Dropout(p=0.3),
            nn.Linear(256, num_features)
        )

        self.classifier = nn.Sequential(
            # nn.Dropout(p=0.3),
            nn.Linear(num_features, 32),
            nn.ReLU(),
#             nn.Dropout(p=0.3),
            nn.Linear(32, num_classes)
        )
        # -----model initialization----- #
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        # -----feature embedding----- #
        x = x.view(-1, self.input_size)
        x_e = self.encoder(x)
        tmp = self.classifier[0](x_e)
        self.hidden = self.classifier[1](tmp)
        y_logit = self.classifier(x_e)

        return x_e, y_logit
    

class Stage_One_Training():
    
    def __init__(self, args, x_labelled_anomaly, x_unlabelled, y_unlabelled, x_val, y_val):
    
        self.num_centroid = args.num_centroid
        self.batch = args.stage_1_batch
        self.epochs = args.stage_1_epochs
        self.kmeans_batch = args.kmeans_batch
        self.feature_dimension = args.feature_dimension
        self.embedded_dimension = args.embedded_dimension
        self.filter = args.filter
        
        self.X_unlabelled = x_unlabelled
        self.X_labelled_anomaly = x_labelled_anomaly
        self.Y_unlabelled = y_unlabelled
        
        self.X_val = x_val
        self.Y_val = y_val
        
        self.seed = 42
        self.lr = args.stage_1_lr
        self.weight_decay = 1e-6
        self.lr_milestones :tuple = ()
        
        # clustering
        kmean = MiniBatchKMeans(n_clusters = self.num_centroid, n_init = self.seed, batch_size = self.kmeans_batch)
        kmean.fit(self.X_unlabelled)
        Y_unlabelled_category = kmean.predict(self.X_unlabelled)
        self.Y_unlabelled_category = Y_unlabelled_category
        
    def shuffle_u(X, Y):
        random_index = np.random.permutation(X.shape[0])
        return X[random_index], Y[random_index]
          
    def train(self):
        
        AE_models = []
        logger.info('Starting Stage_One...') 
        # 以下操作均针对一个簇中的数据
        for k in range(self.num_centroid):
            
            # x_all求的是每个簇中的样本 + labeled anomailes
            x_all = np.vstack((self.X_unlabelled[self.Y_unlabelled_category == k], self.X_labelled_anomaly))
            # 将unlabeled data标记为1
            y_all = np.ones((x_all.shape[0], 1))
            # 将后300个已知异常的标签记为-1
            y_all[-(self.X_labelled_anomaly.shape[0]):] = -1
            # 进行shuffle打乱顺序
            x_all, y_all = shuffle_u(x_all, y_all)
            autoencoder = AutoEncoder(self.feature_dimension, self.embedded_dimension)
            # optimizer = torch.optim.Adam(autoencoder.parameters(),lr=self.lr)
            optimizer = torch.optim.Adam(autoencoder.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)
            
            loss_func = nn.MSELoss()

            stage1_loss = []
            
            # 先对一个簇中的样本进行计算，每个簇运行epoch次，然后再对下一个簇进行计算
            for epoch in range(self.epochs):
                # scheduler.step()
                loss_all = 0.0
                # 一个epoch中所含batch的个数，共有32个batch
                stage1_nbBatch = x_all.shape[0] // self.batch
                autoencoder.train()
                for i in range(stage1_nbBatch):
                    # 选出每个簇中的一个batch_size样本
                    x = x_all[i * self.batch: (i + 1) * self.batch]
                    x = torch.tensor(x).float()
                    # 得到表征和重构后的样本
                    x_e, x_de = autoencoder(x)
                    y = y_all[i * self.batch: (i + 1) * self.batch]
                    y = torch.tensor(y).float()

                    # x.shape = x_de.shape = [256, 1, 196]
                    # 256个(1,196),二维数组的行和列
                    if x.shape != x_de.shape:
                        x = np.reshape(x.data.cpu().numpy(), x_de.shape)
                        x = torch.tensor(x)

                    # 一个样本的每一个维度相减，所以shape = [256, 1, 196]
                    # reduction='none' : (x-x_de)^2
                    # (1,196)表示x和x_de每个维度之间的均方差误差
                    objective_1 = nn.functional.mse_loss(x, x_de, reduction='none')
                    # 求和求出样本与样本之间的重构误差，其shape = (256,1)
                    objective_1 = torch.sum(objective_1, dim=2)
                    # 幂运算，若为1则为本身，若为-1即已知异常则为其倒数
                    objective_1 = objective_1 ** y
                    # 每个batch的重构误差
                    loss = torch.mean(objective_1)
                    loss.requires_grad_(True)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # loss_all该epoch中整个batch的loss之和，即每一个簇中的重构误差
                    # 该loss_all在每个epoch中不断地更新
                    loss_all += loss
                    
                # 这里的loss_all是整个epoch的loss,该epoch是由32个batch累加得到，最终求得的是batch的平均loss
                stage1_loss.append(loss_all / stage1_nbBatch)
                logger.info('第{}个簇\t Epoch {}/{}\t loss: {:.2f}\t'.format(k,epoch+1,self.epochs,loss_all / stage1_nbBatch))
                
            # 保存的是最后一个epoch训练好的AE
            AE_models.append(autoencoder)
                       
        return AE_models
      
    def test(self):
        autoencoders = self.train()                 
        recon = torch.zeros_like(torch.tensor(self.Y_unlabelled_category)).float()
        dist = torch.zeros_like(torch.tensor(self.Y_unlabelled_category)).float()
        # 对每个簇进行操作
        for k in range(self.num_centroid):
            autoencoders[k].eval()
            
            x_unlabelled = torch.tensor(self.X_unlabelled[self.Y_unlabelled_category == k]).float()
            x_unlabelled_e, x_unlabelled_de = autoencoders[k](x_unlabelled)

            # 测试labelled anomaly
            x_labelled = torch.tensor(self.X_labelled_anomaly).float()
            x_labelled_e, x_labelled_de = autoencoders[k](x_labelled)

            if x_unlabelled.shape != x_unlabelled_de.shape:
                x_unlabelled = np.reshape(x_unlabelled.data.cpu().numpy(), x_unlabelled_de.shape)
                x_unlabelled = torch.tensor(x_unlabelled)
                
            self_reconstruction_loss = nn.functional.mse_loss(x_unlabelled, x_unlabelled_de, reduction='none')
            self_reconstruction_loss = torch.sum(self_reconstruction_loss, dim=2)
            # data与重构data的loss
            self_reconstruction_loss = torch.reshape(self_reconstruction_loss, (self_reconstruction_loss.shape[0],))

            # 用于存放每个unlabeled_e与300个labelled_e的最小距离
            tmp = torch.zeros(x_unlabelled_e.shape[0], )

            for i in range(x_unlabelled_e.shape[0]):
                # 计算样本间的欧式距离
                # x_unlabelled_e[i]表示unlabeled data的表征, 每个有64维
                # x_labelled_eb表示labeled anomaly的表征大小为(300,64)
                # 这里计算的是unlabeled_e与每一个labelled_e之间的欧式距离
                # 对于每一个unlabeled_e都有300个距离
                distance = nn.functional.pairwise_distance(x_unlabelled_e[i], x_labelled_e, p=2)
                # 取出300个距离中最小的距离
                tmp[i] = torch.min(distance)

            # 存储对应聚类标签的loss
            # 针对第一个簇，将loss传入属于该簇的list中
            recon[self.Y_unlabelled_category == k] = self_reconstruction_loss
            # 存储对应聚类标签的最小距离
            dist[self.Y_unlabelled_category == k] = tmp
            
        return recon, dist
    
    def filtering(self):
        recon, _ = self.test()
        # 用重构误差作为异常分数, 去掉梯度的形式
        unlabelled_scores = recon.clone().detach().numpy()
        unlabelled_scores = torch.tensor(unlabelled_scores).float()
        # topk返回的是排名前k的值与对应的下标
        # 这里选取异常分数前5%的作为潜在异常
        scores, indexs = unlabelled_scores.topk(int(unlabelled_scores.shape[0] * self.filter), largest=True)
        score_delete = unlabelled_scores[indexs.detach().numpy()].detach().numpy()
        
        # 潜在的异常
        X_deleted = self.X_unlabelled[indexs.detach().numpy()]
        Y_deleted = self.Y_unlabelled[indexs.detach().numpy(), 0]
        # print("潜在异常共有:",Y_deleted.shape[0])
        # print("潜在异常的标签:",Counter(Y_deleted))
        
        # 过滤后的可靠的正常
        reliable_data = list(i for i in range(len(unlabelled_scores)) if i not in indexs)
        X_filter =  self.X_unlabelled[reliable_data]
        Y_filter = self.Y_unlabelled_category[reliable_data]
        # Y_filter_true = self.Y_unlabelled[reliable_data, 0]
        # print("过滤后的可靠正常共有:",Y_filter_true.shape[0])
        # print("过滤后的可靠正常的真实标签:",Counter(Y_filter_true))
        
        return X_deleted, X_filter, Y_deleted, Y_filter, score_delete
    

def calculate_entropy(probs):
    ent = -np.sum(probs * np.log(probs + 1e-8))
    return ent

def calculate_energy(logits):
    energy = torch.logsumexp(logits, dim = 1)
    return energy

def calculate_discrepancy(logits):
    #输入：[n x f]
    #输出：[n x f]
    energy_discrepancy = []
    energy = torch.logsumexp(logits, dim = 1)
    for i, i_logit in enumerate(logits):
        i_logit = i_logit.cpu().detach().numpy()
        i_logit = np.array(i_logit, dtype = np.float128)
        delete_max = np.log(np.sum(np.exp(i_logit)) - np.max(np.exp(i_logit)))
        energy_discrepancy.append(energy[i].cpu().detach().numpy()-delete_max)
    return energy_discrepancy


class Stage_Two_Training():
    def __init__(self, args, x_labelled_anomaly, y_labelled_anomaly, x_deleted, x_filter, 
                 y_deleted, y_filter, score_delete, x_test, y_test, x_val, y_val, X_unlabelled, Y_unlabelled):

        self.batch = args.stage_2_batch
        self.anomaly_batch = args.anomaly_batch
        self.ood_batch = args.ood_batch
        self.epochs = args.stage_2_epochs
        self.loss_oe = args.loss_oe
        self.loss_re = args.loss_re
        self.num_centroid = args.num_centroid
        self.num_anomaly_classes = args.num_anomaly_classes
        self.feature_dimension = args.feature_dimension
        self.embedded_dimension = args.embedded_dimension
        self.num_subgroups = self.num_centroid + self.num_anomaly_classes

        self.X_labelled_anomaly = x_labelled_anomaly
        self.Y_labelled_anomaly = y_labelled_anomaly
        self.X_filter = x_filter
        self.X_deleted = x_deleted
        self.Y_filter = y_filter
        self.Y_deleted = y_deleted
        self.score_delete = score_delete
        
        self.X_test = x_test
        self.Y_test = y_test
        
        self.X_val = x_val
        self.Y_val = y_val
      
        self.weight_decay = 1e-6
        self.lr_milestones : tuple = ()
        self.lr = args.stage_2_lr
        
        self.X_unlabelled = X_unlabelled
        self.Y_unlabelled = Y_unlabelled
       
    def train(self):
        
        # 得到已知异常的batch
        gen_anomaly = getBatch(self.X_labelled_anomaly, self.Y_labelled_anomaly, args.anomaly_batch)
        
        # 真实过滤后的潜在异常是有大部分已知异常、未知异常以及困难正常
        # 若将过滤后的潜在异常全部作为ood，则不合适，所以需要设置权重，对于已知异常权重小，
        ood_data = self.X_deleted
        ood_data_y_true = self.Y_deleted
        # 分数越大，越异常越有可能是属于已知异常，权重越小
        # 权重初始化,使权重在(0,1)之间
        ood_data_w =  (np.max(self.score_delete) - self.score_delete) / (np.max(self.score_delete) - np.min(self.score_delete))

        # 簇(过滤后的可靠正常)+已知异常的标签
        # 希望过滤后的可靠正常为0，未知异常对应于已知异常为1/已知异常的类
        ood_data_y = np.hstack((np.zeros((ood_data.shape[0], self.num_centroid)), np.ones((ood_data.shape[0], self.num_anomaly_classes)) * (1.0 / self.num_anomaly_classes)))
        
        # 返回一个batch_size的ood数据
        gen_ood = getBatchWeigt(ood_data, ood_data_y, ood_data_w, self.ood_batch)
        
        logger.info('Starting Stage_Two...')

        classifier = Classifier(self.feature_dimension, self.embedded_dimension, self.num_subgroups)     
        optimizer = torch.optim.Adam(classifier.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for epoch in range(self.epochs):
            # scheduler.step()
            # 过滤后的可靠正常
            stage2_nbBatch = self.X_filter.shape[0] // self.batch
            X_filter, Y_filter = shuffle_u(self.X_filter, self.Y_filter)
            classifier.train()

            Loss_all=0.0

            for i in range(stage2_nbBatch):

                classifier.train()
                # 过滤后的可靠正常样本(0~4)，其batch_size为128
                x_normal = self.X_filter[i * self.batch: (i + 1) * self.batch]
                x_normal = torch.tensor(x_normal).float()
                y_normal = self.Y_filter[i * self.batch: (i + 1) * self.batch]
                y_normal = torch.tensor(y_normal).to(dtype=torch.int64)
                x_normal_e, logit_normal = classifier(x_normal)

                # 已知异常, 其batch_size = 32
                x_vandal, y_vandal = gen_anomaly.__next__()
                x_vandal = torch.tensor(x_vandal).float()
                y_vandal = np.reshape(y_vandal, y_vandal.shape[0])
                y_vandal = torch.tensor(y_vandal).to(dtype=torch.int64)


                # 将过滤后的可靠正常和已知异常合并
                # 这里的标签为0～8
                x = torch.cat((x_normal, x_vandal), 0)
                y = torch.cat((y_normal, y_vandal), 0)
                x, y = shuffle_u(x, y)
                x_e, y_pred = classifier(x)

                # CrossEntropy Loss
                CELoss = nn.functional.cross_entropy(y_pred, y)

                # Regularization Loss:使模型的置信度变高
                y_pred = y_pred.view(y_pred.size(0), y_pred.size(1), -1)
                y_pred = F.softmax(y_pred, 1)
                regularityLoss = torch.mean(torch.mean(torch.sum(-y_pred * torch.log(y_pred + 1e-8), 1), 1))

                # 过滤后的潜在异常,其batch_size为32
                # 每一个ood的标签都是(0,0,0,0,0,1/3,1/3,1/3)
                x_ood, y_ood, w_ood = gen_ood.__next__()
                x_ood = torch.tensor(x_ood).float()
                y_ood = torch.tensor(y_ood)
                w_ood = torch.tensor(w_ood).float()

                x_ood_e, logits_oe = classifier(x_ood)


                # 相当于监督Loss
                loss_oe = torch.mul(soft_cross_entropy(logits_oe, y_ood, reduction = "none"), w_ood).mean()
                # loss_oe = torch.mul(soft_cross_entropy(torch.stack(selected_unknown_entropy), y_ood[:math.ceil(len(sorted_entropy_top) * 0.2)], reduction = "none"), torch.stack(selected_unknown_weight)).mean()
                Loss =  CELoss  + self.loss_oe * loss_oe +  self.loss_re * regularityLoss

                optimizer.zero_grad()
                Loss.backward()
                optimizer.step()

                Loss_all += Loss
            logger.info('Epoch {}/{}\t loss: {:.2f}\t'.format(epoch+1,self.epochs,Loss_all/stage2_nbBatch))

            classifier.eval()
            x_e, y_logit = classifier(torch.tensor(ood_data).float())
            con_confidence = torch.max(F.softmax(y_logit, dim = 1), dim=1)[0].cpu().detach().numpy()
            # con_confidence = torch.max(F.softmax(y_logit, dim = 1)[:,-self.num_anomaly_classes:], dim=1)[0].cpu().detach().numpy()
            ood_data_w =  (np.max(con_confidence) - con_confidence) / (np.max(con_confidence) - np.min(con_confidence))
            gen_ood = getBatchWeigt(ood_data, ood_data_y, ood_data_w, self.ood_batch)           
        
        aucpr, best_threshold = self.val(classifier)
        
        return classifier, best_threshold

    def val(self, classifier):
        classifier.eval()
        _, y_logit = classifier(torch.tensor(self.X_val).float())
        
        # 未知异常不抛出
        y_val = copy.deepcopy(self.Y_val[:,0])  
        # 将已知异常的标签全部置于1，未知异常和正常置为0
        for i in range(self.num_anomaly_classes):
            y_val[(y_val == self.num_centroid + i)] = 1
        y_val[(y_val == -1) | (y_val == -2)] = 0
       
        prob = torch.max(F.softmax(y_logit, dim = 1)[:,-self.num_anomaly_classes:], dim=1)[0].cpu().detach().numpy()
              
         # 利用val data选择阈值
        best_threshold= self.compute_thresholds(classifier,  self.X_val,  self.Y_val)
        logger.info("best_threshold:{}".format(best_threshold))

        # 用阈值给测试集的预测结果重新赋值
        y_pred = []
        probs = F.softmax(y_logit, dim = 1)[:,-self.num_anomaly_classes:]
        logits_anomaly = y_logit[:,-self.num_anomaly_classes:]
        
        probs_normal = F.softmax(y_logit, dim = 1)[:,:self.num_centroid]
        sum_normal_logits = torch.sum(probs_normal, dim=1).cpu().detach().numpy()
        
        for i, sum_logit in enumerate(sum_normal_logits):
            if sum_logit > self.num_centroid/self.num_subgroups:
                pred_label = 0   #正常打0
            else:
                energy = calculate_discrepancy(logits_anomaly[i, -self.num_anomaly_classes:].unsqueeze(0))
                energy = torch.Tensor(energy).squeeze(0)
                if energy > best_threshold: # 已知异常打上1
                    pred_label = 1 
                else:
                    pred_label = -2  # 剩余样本的预测标签
            y_pred.append(pred_label)
        
        aucroc, aucpr= metric(y_true=y_val, y_score=prob)
  
        return aucpr,best_threshold
 
    def compute_thresholds(self, model, inputs, labels):
        
        entropy_all = []

        # 将已知异常的标签赋为1，正常为0，未知异常为-2
        labels_ = copy.deepcopy(labels)  
        for i in range(self.num_anomaly_classes):
            labels_[(labels_ == self.num_centroid + i)] = 1
        labels_[(labels_ == -1)] = 0
        labels_[(labels_ == -2)] = -2
        
        
        with torch.no_grad():

            _, logits = model(torch.tensor(inputs).float())
            # 选出后m维度预测概率的最大值,all
            # probs = F.softmax(logits, dim = 1)[:,-self.num_anomaly_classes:]
            logits_anomaly = logits[:,-self.num_anomaly_classes:]
            
            # for i, _ in enumerate(probs):
            #     ent = calculate_entropy(probs[i,:].detach().numpy())
            #     entropy_all.append(ent)
            
            energy_all = calculate_discrepancy(logits_anomaly[:,:])
            
            # 阈值是关注目标异常，故将正常和未知异常置为0
            target_labels = copy.deepcopy(labels)  
            for i in range(self.num_anomaly_classes):
                target_labels[(target_labels == self.num_centroid + i)] = 1
            target_labels[(target_labels == -1)] = 0
            target_labels[(target_labels == -2)] = 0
            
            # prob_max = torch.max(probs, dim=1)[0]
            
            #前n维正常的和
            probs_normal = F.softmax(logits, dim = 1)[:,:self.num_centroid]
            sum_normal_logits = torch.sum(probs_normal, dim=1).cpu().detach().numpy()
            #删去被预测为正常的，在剩下的样本中找到阈值
            # 通过entropy小于某个阈值，识别目标异常
            new_energy_all = []
            new_target_labels = []
            for i, sum_logit in enumerate(sum_normal_logits):
                if sum_logit <= self.num_centroid/self.num_subgroups:
                    new_energy_all.append(energy_all[i])
                    new_target_labels.append(target_labels[i])
                    
            # eps = 1e-10         
            # reciprocal_entropy_all = [1/(x+eps) for x in new_entropy_all]
                            
            fpr_1, tpr_1, thresholds_1 = roc_curve(new_target_labels, new_energy_all)

            optimal_th_1, optimal_point_1 = Find_Optimal_Cutoff(TPR=tpr_1, FPR=fpr_1, threshold=thresholds_1)
           
        return optimal_th_1  
    
    def test(self):
        classifier, best_threshold= self.train()
        classifier.eval()
                
        # _, y_logit_val = classifier(torch.tensor(self.X_val).float())
        x_e, y_logit = classifier(torch.tensor(self.X_test).float())
               
        # 未知异常不抛出
        y_test = copy.deepcopy(self.Y_test[:,0])  
        # 将已知异常的标签全部置于1，未知异常和正常置为0
        for i in range(self.num_anomaly_classes):
            y_test[(y_test == self.num_centroid + i)] = 1
        y_test[(y_test == -1) | (y_test == -2)] = 0
               
        y_test_new = copy.deepcopy(self.Y_test[:,0])  
        # 将已知异常的标签全部置于1，未知异常-2,正常0
        for i in range(self.num_anomaly_classes):
            y_test_new[(y_test_new == self.num_centroid + i)] = 1
        y_test_new[(y_test_new == -1)] = 0
        y_test_new[(y_test_new == -2)] = -2
           
        # 得到测试集的概率
         
        prob = torch.max(F.softmax(y_logit, dim = 1)[:,-self.num_anomaly_classes:], dim=1)[0].cpu().detach().numpy()

        # 用阈值给测试集的预测结果重新赋值
        y_pred = []
        probs = F.softmax(y_logit, dim = 1)[:,-self.num_anomaly_classes:]
        logits_anomaly = y_logit[:,-self.num_anomaly_classes:]
        
        probs_normal = F.softmax(y_logit, dim = 1)[:,:self.num_centroid]
        sum_normal_logits = torch.sum(probs_normal, dim=1).cpu().detach().numpy()
        
        energy_all = calculate_discrepancy(logits_anomaly[:, -self.num_anomaly_classes:])
        
        for i, sum_logit in enumerate(sum_normal_logits):
            if sum_logit > self.num_centroid/self.num_subgroups:
                pred_label = 0   #正常打0
            else:
                energy = calculate_discrepancy(logits_anomaly[i, -self.num_anomaly_classes:].unsqueeze(0))
                energy = torch.Tensor(energy).squeeze(0)
                # eps = 1e-10         
                # reciprocal_ent = 1/(ent+eps)
                if energy > best_threshold: # 已知异常打上1
                    pred_label = 1 
                else:
                    pred_label = -2  # 剩余样本的预测标签
            y_pred.append(pred_label)
        
        cm_test = confusion_matrix(y_test_new, y_pred, labels = [0,1,-2])    
     
        TP_normal = cm_test[0][0]
        FP_normal = cm_test[1][0] + cm_test[2][0]
        FN_normal = cm_test[0][1] + cm_test[0][2]
        
        precision_normal = TP_normal / (TP_normal + FP_normal)
        recall_normal = TP_normal / (TP_normal + FN_normal)
        f1_normal = 2*(precision_normal*recall_normal)/(precision_normal+recall_normal)
        
        TP_anomaly = cm_test[1][1]
        FP_anomaly = cm_test[0][1] + cm_test[2][1]
        FN_anomaly = cm_test[1][0] + cm_test[1][2]
        
        precision_anomaly = TP_anomaly / (TP_anomaly + FP_anomaly)
        recall_anomaly = TP_anomaly / (TP_anomaly + FN_anomaly)
        f1_anomaly = 2*(precision_anomaly*recall_anomaly)/(precision_anomaly+recall_anomaly)
        
        TP_unknown = cm_test[2][2]
        FP_unknown = cm_test[0][2] + cm_test[1][2]
        FN_unknown = cm_test[2][0] + cm_test[2][1]
        
        precision_unknown = TP_unknown / (TP_unknown + FP_unknown)
        recall_unknown = TP_unknown / (TP_unknown + FN_unknown)
        f1_unknown = 2*(precision_unknown*recall_unknown)/(precision_unknown+recall_unknown)
        
        # 宏平均是直接计算各个类别的平均值，不考虑类别的样本数量。
        macro_avg_precision = (precision_normal + precision_anomaly + precision_unknown) / 3
        macro_avg_recall = (recall_normal + recall_anomaly + recall_unknown) / 3
        macro_avg_f1_score = (f1_normal + f1_anomaly + f1_unknown) / 3
        
        
        # 加权平均是根据每个类别的样本数量进行加权的平均值。
        normal_all = cm_test[0][0] + cm_test[0][1] + cm_test[0][2]
        anomaly_all = cm_test[1][0] + cm_test[1][1] + cm_test[1][2]
        unknown_all = cm_test[2][0] + cm_test[2][1] + cm_test[2][2]
        sample_all = normal_all + anomaly_all + unknown_all
        
        weighted_avg_precision = (precision_normal * normal_all + precision_anomaly * anomaly_all
                                 + precision_unknown * unknown_all)/sample_all
        
        weighted_avg_recall = (recall_normal * normal_all + recall_anomaly * anomaly_all
                                 + recall_unknown * unknown_all)/sample_all
        
        weighted_avg_f1_score = (f1_normal * normal_all + f1_anomaly * anomaly_all
                                 + f1_unknown * unknown_all)/sample_all
            
        aucroc, aucpr = metric(y_true=y_test, y_score=prob)

        return aucroc, aucpr
        
        
def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point

from sklearn.metrics import roc_auc_score, precision_recall_curve,auc
def metric(y_true, y_score, pos_label=1):
    aucroc = roc_auc_score(y_true, y_score)
    precision, recall, threshold = precision_recall_curve(y_true, y_score)
    aucpr = auc(recall, precision)
    return aucroc, aucpr

def fpr_recall(conf, label, tpr):
    vandal_conf = conf[label == 1]
    normal_conf = conf[label == 0]
    num_vandal = len(vandal_conf)
    num_normal = len(normal_conf)
    # recall_num = 误抛的总数 = 误抛出的正常+误抛的未知异常
    recall_num = int(np.floor(tpr * num_vandal))

    thresh = np.sort(vandal_conf)[-recall_num]
    num_fp = np.sum(normal_conf >= thresh)
    fpr = num_fp / num_normal
    return fpr, thresh, recall_num

xp_path = './result'
import os
if not os.path.exists(xp_path):
    os.makedirs(xp_path)
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_file = xp_path + '/log.txt'
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

model_filter = Stage_One_Training(args, X_labelled_anomaly, X_unlabelled, Y_unlabelled, X_val, Y_val)
X_deleted, X_filter, Y_deleted, Y_filter, Score_delete = model_filter.filtering()
model = Stage_Two_Training(args, X_labelled_anomaly,Y_labelled_anomaly, X_deleted, X_filter, 
                       Y_deleted, Y_filter, Score_delete, X_test, Y_test, X_val, Y_val, X_unlabelled, Y_unlabelled)
aucroc, aucpr = model.test()
print(aucroc, aucpr)