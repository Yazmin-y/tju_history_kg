# -*- coding: utf-8 -*-
from django.shortcuts import render
from util.pre_load import neo4jconn

import json


def search_all_events(request):
    entityRelation = neo4jconn.get_all_event()
    return render(request, 'event_search.html', {'entityRelation': json.dumps(entityRelation, ensure_ascii=False)})


def search_all_headmaster(request):
    entityRelation = neo4jconn.get_all_headmaster()
    return render(request, 'headmaster_search.html', {'entityRelation': json.dumps(entityRelation, ensure_ascii=False)})


def search_entity(request):
    ctx = {}
    if request.GET:
        entity = request.GET['user_text']
        entity = entity.strip()
        entity = entity.lower()
        entity = ''.join(entity.split())

        entityRelation = neo4jconn.get_entity_info(entity)
        if len(entityRelation) == 0:
            # 若数据库中无法找到该实体，则返回数据库中无该实体
            ctx = {'title': '<h2>知识库中暂未添加该实体</h1>'}
            return render(request, 'entity_search.html', {'ctx': json.dumps(ctx, ensure_ascii=False)})
        else:
            return render(request, 'entity_search.html',
                          {'entityRelation': json.dumps(entityRelation, ensure_ascii=False)})
    # 需要进行类型转换
    return render(request, 'entity_search.html', {'ctx': ctx})


# 关系查询
def search_relation(request):
    ctx = {}
    if (request.GET):
        # 实体1
        entity1 = request.GET['entity1_text']
        entity1 = entity1.strip()
        entity1 = entity1.lower()
        entity1 = ''.join(entity1.split())

        # 关系
        relation = request.GET['relation_name_text']
        # 将关系名转为大写
        relation = relation.upper()

        # 实体2
        entity2 = request.GET['entity2_text']
        entity2 = entity2.strip()
        entity2 = entity2.lower()
        entity2 = ''.join(entity2.split())

        # 1.若只输入entity1,则输出与entity1有直接关系的实体和关系
        if len(entity1) != 0 and len(relation) == 0 and len(entity2) == 0:
            searchResult = neo4jconn.findRelationByEntity1(entity1)
            if len(searchResult) > 0:
                return render(request, 'relation.html', {'searchResult': json.dumps(searchResult, ensure_ascii=False)})

        # 2.若只输入entity2则,则输出与entity2有直接关系的实体和关系
        if len(entity2) != 0 and len(relation) == 0 and len(entity1) == 0:
            searchResult = neo4jconn.findRelationByEntity2(entity2)
            if len(searchResult) > 0:
                return render(request, 'relation.html', {'searchResult': json.dumps(searchResult, ensure_ascii=False)})

        # 3.若输入entity1和relation，则输出与entity1具有relation关系的其他实体
        if len(entity1) != 0 and len(relation) != 0 and len(entity2) == 0:
            searchResult = neo4jconn.findOtherEntities(entity1, relation)
            if len(searchResult) > 0:
                return render(request, 'relation.html', {'searchResult': json.dumps(searchResult, ensure_ascii=False)})

        # 4.若输入entity2和relation，则输出与entity2具有relation关系的其他实体
        if len(entity2) != 0 and len(relation) != 0 and len(entity1) == 0:
            searchResult = neo4jconn.findOtherEntities2(entity2, relation)
            if len(searchResult) > 0:
                return render(request, 'relation.html', {'searchResult': json.dumps(searchResult, ensure_ascii=False)})

        # 5.若输入entity1和entity2,则输出entity1和entity2之间的关系
        if len(entity1) != 0 and len(relation) == 0 and len(entity2) != 0:
            searchResult = neo4jconn.findRelationByEntities(entity1, entity2)
            if len(searchResult) > 0:
                return render(request, 'relation.html', {'searchResult': json.dumps(searchResult, ensure_ascii=False)})

        # 6.若输入entity1,entity2和relation,则输出entity1、entity2是否具有相应的关系
        if len(entity1) != 0 and len(entity2) != 0 and len(relation) != 0:
            print(relation)
            searchResult = neo4jconn.findEntityRelation(entity1, relation, entity2)
            if len(searchResult) > 0:
                return render(request, 'relation.html', {'searchResult': json.dumps(searchResult, ensure_ascii=False)})

        # 7.若全为空
        if len(entity1) != 0 and len(relation) != 0 and len(entity2) != 0:
            pass

        ctx = {'title': '<h1>暂未找到相应的匹配</h1>'}
        return render(request, 'relation.html', {'ctx': ctx})

    return render(request, 'relation.html', {'ctx': ctx})
