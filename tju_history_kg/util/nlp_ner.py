# -*- coding: utf-8 -*-

'''
人名:nr
地名:ns
机构名:nt

'''


# 获取实体信息
def get_ner_info(flag):
    if flag == 'nr':
        return 'nr'
    if flag == 'ns':
        return 'ns'
    if flag == 'nt':
        return 'nt'


def get_detail_ner_info(flag):
    if flag == 'nr':
        return '人物'
    if flag == 'ns':
        return '地名'
    if flag == 'nt':
        return '机构'


def get_ner(word_pair):
    ner_list = []
    for term in word_pair:
        word = term.word
        pos = str(term.flag)
        if pos.startswith('nr'):
            ner_list.append([word, 'nr'])
        elif pos.startswith('ns'):
            ner_list.append([word, 'ns'])
        elif pos.startswith('nt'):
            ner_list.append([word, 'nt'])
        else:
            ner_list.append([word, 0])
    return ner_list
