# -*- coding:utf-8 -*-
import torch
import torch.nn.functional as F
from django.shortcuts import render
import jieba.posseg as posseg
import jieba
import os
from util.pre_load import neo4jconn, model_dict

model, words, classes, index_classes = model_dict
model.eval()

root = os.path.abspath(os.path.join(os.getcwd(), ".."))
dict_path = os.path.join(root, 'data/custom_dict/user_dict.txt')
jieba.load_userdict(dict_path)


def question_answering(request):
    context = {'ctx': ''}
    if (request.GET):
        question = request.GET['question']
        # question = question[:300]
        # 移除空格
        question = question.strip()
        question = question.lower()

        # word_nature = segment.seg(question)
        # word_pair = posseg.cut(question)
        # print([se for se in word_pair])
        classfication_num = chatbot_response(question)
        print('类别：{}'.format(classfication_num))

        if classfication_num == 0:
            word = ''
            word_pair = posseg.cut(question)
            for term in word_pair:
                if str(term.flag) == 'nt':
                    word = term.word
                    ret_dict = neo4jconn.sub_college(word)
                    print(ret_dict)
                    break
            if word == '':
                ret_dict = []

        elif classfication_num == 1:
            word = ''
            word_pair = posseg.cut(question)
            for term in word_pair:
                if str(term.flag) == 'nt':
                    word = term.word
                    ret_dict = neo4jconn.trans_in(word)
                    break
            if word == '':
                ret_dict = []

        elif classfication_num == 2:
            word = ''
            word_pair = posseg.cut(question)
            for term in word_pair:
                print(term)
                if str(term.flag) == 'nt':
                    word = term.word
                    ret_dict = neo4jconn.trans_out(word)
                    break
            if word == '':
                ret_dict = []

        elif classfication_num == 3:
            word = ''
            word_pair = posseg.cut(question)
            for term in word_pair:
                print(term)
                if str(term.flag) == 'm':
                    word = term.word
                    ret_dict = neo4jconn.leader(word)
                    break
            if word == '':
                ret_dict = []

        elif classfication_num == 4:
            word = ''
            word_pair = posseg.cut(question)
            for term in word_pair:
                print(term)
                if str(term.flag) == 'm':
                    word = term.word
                    ret_dict = neo4jconn.school_name(word)
                    break
            if word == '':
                ret_dict = []

        elif classfication_num == 5:
            word = ''
            word_pair = posseg.cut(question)
            for term in word_pair:
                print(term)
                if str(term.flag) == 'm':
                    word = term.word
                    ret_dict = neo4jconn.event_name(word)
                    break
            if word == '':
                ret_dict = []

        elif classfication_num == 6:
            word = ''
            # print(word_pair)
            word_pair = posseg.cut(question)
            for term in word_pair:
                print(term)
                if str(term.flag) == 'nt':
                    word = term.word
                    ret_dict = neo4jconn.teachers(word)
                    break
            if word == '':
                ret_dict = []

        elif classfication_num == 7:
            word = ''
            # print(word_pair)
            word_pair = posseg.cut(question)
            for term in word_pair:
                print(term)
                if str(term.flag) == 'nt':
                    word = term.word
                    ret_dict = neo4jconn.precursor(word)
                    break
            if word == '':
                ret_dict = []

        elif classfication_num == 8:
            word = ''
            word_pair = posseg.cut(question)
            for term in word_pair:
                print(term)
                if str(term.flag) == 'nt':
                    word = term.word
                    ret_dict = neo4jconn.start_time(word)
                    break
            if word == '':
                ret_dict = []

        elif classfication_num == 9:
            word = ''
            word_pair = posseg.cut(question)
            for term in word_pair:
                print(term)
                if str(term.flag) == 'nt':
                    word = term.word
                    ret_dict = neo4jconn.teacher_cnt(word)
                    break
            if word == '':
                ret_dict = []

        if (len(ret_dict) != 0):
            print(ret_dict)
            return render(request, 'question_answering.html', {'ret': ret_dict})

        return render(request, 'question_answering.html', {'ctx': '暂未找到答案'})

    return render(request, 'question_answering.html', context)


def sentence_segment(sentence):
    word_pair = posseg.cut(sentence)
    sentence_words = []
    for term in word_pair:
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
    predicted_prob, predicted_index = torch.max(F.softmax(outputs, 1), 1)  # 预测最大类别的概率与索引
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
