# -*- coding: utf-8 -*-
from Model.neo4j_models import Neo4j_Handle

import torch
import torch.nn as nn
import pickle
import os


# 初始化模型
def init_model():
    words_path = os.path.join(os.getcwd()+'/util', "words.pkl")
    with open(words_path, 'rb') as f_words:
        words = pickle.load(f_words)

    classes_path = os.path.join(os.getcwd()+'/util', "classes.pkl")
    with open(classes_path, 'rb') as f_classes:
        classes = pickle.load(f_classes)

    classes_index_path = os.path.join(os.getcwd()+'/util', "classes_index.pkl")
    with open(classes_index_path, 'rb') as f_classes_index:
        classes_index = pickle.load(f_classes_index)

    index_classes = dict(zip(classes_index.keys(), classes_index.values()))

    print('index_classes:{}'.format(index_classes))

    class classifyModel(nn.Module):

        def __init__(self):
            super(classifyModel, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(len(words), 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(64, len(classes)))

        def forward(self, x):
            out = self.model(x)
            return out

    model = classifyModel()
    model_path = os.path.join(os.getcwd()+'/util', "model.h5")
    pretrained = torch.load(model_path)
    model.load_state_dict(pretrained)
    return model, words, classes, index_classes


# 初始化neo4j
def init_neo4j():
    neo4jconn = Neo4j_Handle()
    neo4jconn.connectNeo4j()
    return neo4jconn


# 初始化
neo4jconn = init_neo4j()

# 初始化分类模型，词典等
model_dict = init_model()
