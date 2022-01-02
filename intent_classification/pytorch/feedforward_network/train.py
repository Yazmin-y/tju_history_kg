# -*- coding: utf-8 -*-

import pickle
import numpy as np
import os
import random
import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

intent_classification_path = os.path.abspath(os.path.join(os.getcwd(), '../..'))

# 训练数据路径
train_data = os.path.join(intent_classification_path, 'classification_data/classification_data.txt')

# 类别与索引号
class_index_data = os.path.join(intent_classification_path, 'classification_data/question_class.txt')

# 所有不同单词
words = []
# 所有类别
classes = []
# 类别对应的索引号
classes_index = {}
# 所有文档
documents = []

with open(class_index_data, 'r', encoding='utf-8') as f_read:
    for line in f_read:
        line = line.strip()
        tokens = line.split(":")
        classes.append(tokens[1])
        classes_index[int(tokens[0])] = tokens[1]

with open(train_data, 'r', encoding='utf-8') as f_read:
    for line in f_read:
        line = line.strip()
        tokens = line.split(',')
        doc_words = tokens[1].split(' ')
        words.extend(doc_words)
        documents.append((doc_words, int(tokens[0])))

words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

print('classes_index:{}'.format(classes_index))
print("####################")
print(len(documents), "documents")
print("####################")
print(len(classes), "classes", classes)
print("####################")
print(len(words), "unique words", words)

# 保存相关数据
words_path = os.path.join(os.getcwd(), 'words.pkl')
classes_path = os.path.join(os.getcwd(), 'classes.pkl')
classes_index_path = os.path.join(os.getcwd(), 'classes_index.pkl')
with open(words_path, 'wb') as f_words, open(classes_path, 'wb') as f_classes, open(classes_index_path,
                                                                                    'wb') as f_classes_index:
    pickle.dump(words, f_words)
    pickle.dump(classes, f_classes)
    pickle.dump(classes_index, f_classes_index)
    print('save data done!')

training = []
for doc in documents:
    # 词袋
    line_words = doc[0]  # 文档的词
    bag = [0] * len(words)
    for s in line_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1  # 词在词典中
    training.append([bag, doc[1]])
random.shuffle(training)
training = np.array(training)
train_doc = list(training[:, 0])
train_target = list(training[:, 1])
print("{},\n\n {}".format(train_doc, train_target))

writer = SummaryWriter(os.getcwd() + '/log', comment='feedforward_network')

print('train_doc len:{}'.format(len(train_doc)))
print('train_target len:{}'.format(len(classes)))


class classifyModel(nn.Module):

    def __init__(self):
        super(classifyModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(len(train_doc[4]), 128),
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

optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True)
criterion = nn.CrossEntropyLoss()

train_doc = torch.tensor(train_doc)
train_doc = train_doc.float()
train_target = torch.tensor(train_target)
# train_target = train_target.long()

print('{},{}'.format(train_doc.dtype, train_target.dtype))

losses = []
for iter in range(300):
    out = model(train_doc)
    loss = criterion(out, train_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (iter + 1) % 10 == 0:
        losses.append(loss.item())
        print('iter [{}/{}], Loss: {:.4f}'.format(iter + 1, 300, loss.item()))
    writer.add_graph(model, input_to_model=train_doc, verbose=False)
    writer.add_scalar('loss', loss.item(), global_step=iter + 1)
plt.plot(losses)
plt.show()
writer.flush()
writer.close()

model_path = os.path.join(os.getcwd(), "model.h5")
torch.save(model.state_dict(), model_path)
