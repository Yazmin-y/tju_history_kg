import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os
import jieba.posseg as posseg
import jieba

root = os.path.abspath(os.path.join(os.getcwd(), "../../../"))
dict_path = os.path.join(root, 'data/custom_dict/user_dict.txt')
jieba.load_userdict(dict_path)

words_path = os.path.join(os.getcwd(), "words.pkl")
with open(words_path, 'rb') as f_words:
    words = pickle.load(f_words)

classes_path = os.path.join(os.getcwd(), "classes.pkl")
with open(classes_path, 'rb') as f_classes:
    classes = pickle.load(f_classes)

classes_index_path = os.path.join(os.getcwd(), "classes_index.pkl")
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
model_path = os.path.join(os.getcwd(), "model.h5")
model.load_state_dict(torch.load(model_path))


def sentence_segment(sentence):
    word_nature = posseg.cut(sentence)
    print(word_nature)
    sentence_words = []
    for term in word_nature:
        if str(term.flag) == 'nr':
            sentence_words.append('nr')
        elif str(term.flag) == 'nt':
            sentence_words.append('nt')
        elif str(term.flag) == 't':
            sentence_words.append('t')
        elif str(term.flag) == 'm':
            sentence_words.append('x')
        else:
            sentence_words.append(term.word)
    print(sentence_words)
    return sentence_words


def bow(sentence, words, show_detail=True):
    sentence_words = sentence_segment(sentence)
    # 词袋
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1  # 词在词典中
            if show_detail:
                print("found in bag:{}".format(w))
    return [bag]


def predict_class(sentence, model):
    sentence_bag = bow(sentence, words, False)
    model.eval()
    with torch.no_grad():
        outputs = model(torch.FloatTensor(sentence_bag))
    print('outputs:{}'.format(outputs))
    predicted_prob, predicted_index = torch.max(F.softmax(outputs, 1), 1)  # 预测最大类别的概率与索引
    print('softmax_prob:{}'.format(predicted_prob))
    print('softmax_index:{}'.format(predicted_index))
    results = []
    # results.append({'intent':index_classes[predicted_index.detach().numpy()[0]], 'prob':predicted_prob.detach().numpy()[0]})
    results.append({'intent': predicted_index.detach().numpy()[0], 'prob': predicted_prob.detach().numpy()[0]})
    print('result:{}'.format(results))
    return results


def get_response(predict_result):
    tag = predict_result[0]['intent']
    return tag


def chatbot_response(text):
    predict_result = predict_class(text, model)
    res = get_response(predict_result)
    return res


print(chatbot_response("智能与计算学部的老师有谁？"))