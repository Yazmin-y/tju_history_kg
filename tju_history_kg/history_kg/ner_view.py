# -*- coding: utf-8 -*-
from django.shortcuts import render

# import sys
# sys.path.append("..")
# from util.pre_load import segment
import jieba.posseg as pseg
from util.nlp_ner import get_ner, get_ner_info, get_detail_ner_info


# 分词+词性+实体识别
def ner_post(request):
    ctx = {}
    if request.POST:
        # 获取输入文本
        input_text = request.POST['user_text']

        input_text = input_text[:300]
        # 移除空格
        input_text = input_text.strip()
        # 分词
        # word_nature = segment.seg(input_text)
        word_pair = pseg.cut(input_text)

        text = ""
        # 实体识别
        ner_list = get_ner(word_pair)
        # 遍历输出
        for pair in ner_list:
            if pair[1] == 0:
                text += pair[0]
                continue
            # text += "<a href='detail.html?title=" + pair[0] + "'  data-original-title='" + get_ner_info(pair[1]) + "'  data-placement='top' data-trigger='hover' data-content='"+get_detail_ner_info(pair[1])+"' class='popovers'>" + pair[0] + "</a>"
            if pair[1] == 'nr':  # 人物
                text += "<a style='color:blue'   data-original-title='" + get_ner_info(
                    pair[1]) + "'  data-placement='top' data-trigger='hover' data-content='" + get_detail_ner_info(
                    pair[1]) + "' class='popovers'>" + pair[0] + "</a>"
            elif pair[1] == 'ns':  # 地名
                text += "<a style='color:red'  data-original-title='" + get_ner_info(
                    pair[1]) + "'  data-placement='top' data-trigger='hover' data-content='" + get_detail_ner_info(
                    pair[1]) + "' class='popovers'>" + pair[0] + "</a>"
            elif pair[1] == 'nt':  # 机构
                text += "<a style='color:orange'  data-original-title='" + get_ner_info(
                    pair[1]) + "'  data-placement='top' data-trigger='hover' data-content='" + get_detail_ner_info(
                    pair[1]) + "' class='popovers'>" + pair[0] + "</a>"

        ctx['rlt'] = text

        # 获取词和词性
        seg_word = list(
            term.word + " <strong><small>[" + str(term.nature) + "]</small></strong> " for term in word_pair)
        seg_word = ''.join(seg_word)
        ctx['seg_word'] = seg_word

    return render(request, "entity_reg.html", ctx)
