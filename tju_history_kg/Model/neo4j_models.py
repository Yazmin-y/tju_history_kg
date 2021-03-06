# -*- coding: utf-8 -*-
from py2neo import Graph, NodeMatcher


class Neo4j_Handle():
    graph = None
    matcher = None

    def __init__(self):
        print("Neo4j Init ...")

    def connectNeo4j(self):
        self.graph = Graph("http://127.0.0.1:7474", auth=('neo4j', '123123'))
        self.matcher = NodeMatcher(self.graph)

    def get_all_event(self):
        data = self.graph.run(
            "match (source:event)-[rel]-(target)" +
            "return rel ").data()

        json_list = []
        for an in data:
            result = {}
            rel = an['rel']
            start_name = rel.start_node['name']
            start_intro = rel.start_node['introduction']
            result['source'] = {'name': start_name,
                                'time': rel.start_node['time'],
                                'type': str(rel.start_node.labels),
                                'intro': start_intro,
                                'image': rel.start_node['image']}

            json_list.append(result)
        return json_list

    def get_all_headmaster(self):
        data = self.graph.run(
            "match (source:headmaster)-[rel]-(target)" +
            "return rel ").data()

        json_list = []
        for an in data:
            result = {}
            rel = an['rel']
            relation_type = list(rel.types())[0]
            start_name = rel.start_node['name']
            start_intro = rel.start_node['introduction']
            end_name = rel.end_node['name']
            end_location = rel.end_node['location']
            result['source'] = {'name': start_name, 'start_time': rel.start_node['start_time'], 'end_time': rel.start_node['end_time'],
                                'type': str(rel.start_node.labels),
                                'intro': start_intro, 'title': rel.start_node['title'],
                                'birth_year': rel.start_node['birthyear'],
                                'death_year': rel.start_node['deathyear'],
                                'image': rel.start_node['image']}
            result['target'] = {'name': end_name, 'type': str(rel.end_node.labels),
                                'location': end_location,
                                'start_time': rel.end_node['start_time'],
                                'end_time': rel.end_node['end_time']}
            result['rel_type'] = relation_type

            json_list.append(result)
        return json_list

    def get_entity_info(self, name):
        data = self.graph.run(
            "match (source)-[rel]-(target) where source.name = $name " +
            "return rel ", name=name).data()

        json_list = []
        for an in data:
            result = {}
            rel = an['rel']
            relation_type = list(rel.types())[0]
            start_name = rel.start_node['name']
            start_intro = rel.start_node['introduction']
            end_name = rel.end_node['name']
            end_intro = rel.end_node['introduction']
            result['source'] = {'name': start_name, 'time': rel.start_node['time'],
                                'type': str(rel.start_node.labels),
                                'intro': start_intro,
                                'title': rel.start_node['title'],
                                'image': rel.start_node['image'] or rel.start_node['imageURL'],
                                'website': rel.start_node['website'] or rel.start_node['homepage'],
                                'birth_year': rel.start_node['birthyear'],
                                'death_year': rel.start_node['deathyear'],
                                'start_time': rel.start_node['start_time'],
                                'end_time': rel.start_node['end_time'],
                                'email': rel.start_node['email'],
                                'direction': rel.start_node['direction']
                                }
            result['target'] = {'name': end_name, 'time': rel.end_node['time'],
                                'type': str(rel.end_node.labels),
                                'intro': end_intro,
                                'title': rel.end_node['title'],
                                'image': rel.end_node['image'] or rel.end_node['imageURL'],
                                'website': rel.end_node['website'] or rel.start_node['homepage'],
                                'birth_year': rel.end_node['birthyear'],
                                'death_year': rel.end_node['deathyear'],
                                'start_time': rel.end_node['start_time'],
                                'end_time': rel.end_node['end_time'],
                                'email': rel.end_node['email'],
                                'direction': rel.end_node['direction']
                                }
            result['rel_type'] = relation_type

            json_list.append(result)
        return json_list

    # ???.????????????????????????1?????????
    # 1.????????????:??????1(?????????1?????????????????????????????????)
    def findRelationByEntity1(self, entity1):
        answer = self.graph.run(
            "match (source)-[rel]-(target)  where source.name = $name " +
            "return rel ", name=entity1).data()

        answer_list = []
        for an in answer:
            result = {}
            rel = an['rel']
            relation_type = list(rel.types())[0]
            start_name = rel.start_node['name']
            end_name = rel.end_node['name']
            result["source"] = {'name': start_name}
            result['type'] = relation_type
            result['target'] = {'name': end_name}
            answer_list.append(result)

        return answer_list

    # 2.?????????????????????2
    def findRelationByEntity2(self, entity1):
        answer = self.graph.run(
            "match (source)-[rel]-(target)  where target.name = $name " +
            "return rel ", name=entity1).data()
        json_list = []
        for an in answer:
            result = {}
            rel = an['rel']
            relation_type = list(rel.types())[0]
            start_name = rel.start_node['name']
            start_intro = rel.start_node['introduction']
            end_name = rel.end_node['name']
            end_intro = rel.end_node['introduction']
            result['source'] = {'name': start_name, 'time': rel.start_node['time'],
                                'type': str(rel.start_node.labels),
                                'intro': start_intro,
                                'title': rel.start_node['title'],
                                'image': rel.start_node['image'] or rel.start_node['imageURL'],
                                'website': rel.start_node['website'] or rel.start_node['homepage'],
                                'birth_year': rel.start_node['birthyear'],
                                'death_year': rel.start_node['deathyear'],
                                'start_time': rel.start_node['start_time'],
                                'end_time': rel.start_node['end_time'],
                                'email': rel.start_node['email'],
                                'direction': rel.start_node['direction']
                                }
            result['target'] = {'name': end_name, 'time': rel.end_node['time'],
                                'type': str(rel.end_node.labels),
                                'intro': end_intro,
                                'title': rel.end_node['title'],
                                'image': rel.end_node['image'] or rel.end_node['imageURL'],
                                'website': rel.end_node['website'] or rel.start_node['homepage'],
                                'birth_year': rel.end_node['birthyear'],
                                'death_year': rel.end_node['deathyear'],
                                'start_time': rel.end_node['start_time'],
                                'end_time': rel.end_node['end_time'],
                                'email': rel.end_node['email'],
                                'direction': rel.end_node['direction']
                                }
            result['rel_type'] = relation_type
            json_list.append(result)
        return json_list

    # 3.?????????????????????1+??????
    def findOtherEntities(self, entity1, relation):
        answer = self.graph.run(
            "match (source)-[rel:" + relation + "]->(target)  where source.name = $name " +
            "return rel ", name=entity1).data()

        json_list = []
        for an in answer:
            result = {}
            rel = an['rel']
            relation_type = list(rel.types())[0]
            start_name = rel.start_node['name']
            start_intro = rel.start_node['introduction']
            end_name = rel.end_node['name']
            end_intro = rel.end_node['introduction']
            result['source'] = {'name': start_name, 'time': rel.start_node['time'],
                                'type': str(rel.start_node.labels),
                                'intro': start_intro,
                                'title': rel.start_node['title'],
                                'image': rel.start_node['image'] or rel.start_node['imageURL'],
                                'website': rel.start_node['website'] or rel.start_node['homepage'],
                                'birth_year': rel.start_node['birthyear'],
                                'death_year': rel.start_node['deathyear'],
                                'start_time': rel.start_node['start_time'],
                                'end_time': rel.start_node['end_time'],
                                'email': rel.start_node['email'],
                                'direction': rel.start_node['direction']
                                }
            result['target'] = {'name': end_name, 'time': rel.end_node['time'],
                                'type': str(rel.end_node.labels),
                                'intro': end_intro,
                                'title': rel.end_node['title'],
                                'image': rel.end_node['image'] or rel.end_node['imageURL'],
                                'website': rel.end_node['website'] or rel.start_node['homepage'],
                                'birth_year': rel.end_node['birthyear'],
                                'death_year': rel.end_node['deathyear'],
                                'start_time': rel.end_node['start_time'],
                                'end_time': rel.end_node['end_time'],
                                'email': rel.end_node['email'],
                                'direction': rel.end_node['direction']
                                }
            result['rel_type'] = relation_type
            json_list.append(result)
        return json_list

    # 4.?????????????????????+??????2
    def findOtherEntities2(self, entity2, relation):

        answer = self.graph.run(
            "match (source)-[rel:" + relation + "]->(target)  where target.name = $name " +
            "return rel ", name=entity2).data()

        json_list = []
        for an in answer:
            result = {}
            rel = an['rel']
            relation_type = list(rel.types())[0]
            start_name = rel.start_node['name']
            start_intro = rel.start_node['introduction']
            end_name = rel.end_node['name']
            end_intro = rel.end_node['introduction']
            result['source'] = {'name': start_name, 'time': rel.start_node['time'],
                                'type': str(rel.start_node.labels),
                                'intro': start_intro,
                                'title': rel.start_node['title'],
                                'image': rel.start_node['image'] or rel.start_node['imageURL'],
                                'website': rel.start_node['website'] or rel.start_node['homepage'],
                                'birth_year': rel.start_node['birthyear'],
                                'death_year': rel.start_node['deathyear'],
                                'start_time': rel.start_node['start_time'],
                                'end_time': rel.start_node['end_time'],
                                'email': rel.start_node['email'],
                                'direction': rel.start_node['direction']
                                }
            result['target'] = {'name': end_name, 'time': rel.end_node['time'],
                                'type': str(rel.end_node.labels),
                                'intro': end_intro,
                                'title': rel.end_node['title'],
                                'image': rel.end_node['image'] or rel.end_node['imageURL'],
                                'website': rel.end_node['website'] or rel.start_node['homepage'],
                                'birth_year': rel.end_node['birthyear'],
                                'death_year': rel.end_node['deathyear'],
                                'start_time': rel.end_node['start_time'],
                                'end_time': rel.end_node['end_time'],
                                'email': rel.end_node['email'],
                                'direction': rel.end_node['direction']
                                }
            result['rel_type'] = relation_type
            json_list.append(result)
        return json_list

    # 5.?????????????????????1+??????2
    def findRelationByEntities(self, entity1, entity2):
        answer = self.graph.run(
            "match (source)-[rel]-(target)  where source.name= $name1 and target.name = $name2 " +
            "return rel ", name1=entity1, name2=entity2).data()

        json_list = []
        for an in answer:
            result = {}
            rel = an['rel']
            relation_type = list(rel.types())[0]
            start_name = rel.start_node['name']
            start_intro = rel.start_node['introduction']
            end_name = rel.end_node['name']
            end_intro = rel.end_node['introduction']
            result['source'] = {'name': start_name, 'time': rel.start_node['time'],
                                'type': str(rel.start_node.labels),
                                'intro': start_intro,
                                'title': rel.start_node['title'],
                                'image': rel.start_node['image'] or rel.start_node['imageURL'],
                                'website': rel.start_node['website'] or rel.start_node['homepage'],
                                'birth_year': rel.start_node['birthyear'],
                                'death_year': rel.start_node['deathyear'],
                                'start_time': rel.start_node['start_time'],
                                'end_time': rel.start_node['end_time'],
                                'email': rel.start_node['email'],
                                'direction': rel.start_node['direction']
                                }
            result['target'] = {'name': end_name, 'time': rel.end_node['time'],
                                'type': str(rel.end_node.labels),
                                'intro': end_intro,
                                'title': rel.end_node['title'],
                                'image': rel.end_node['image'] or rel.end_node['imageURL'],
                                'website': rel.end_node['website'] or rel.start_node['homepage'],
                                'birth_year': rel.end_node['birthyear'],
                                'death_year': rel.end_node['deathyear'],
                                'start_time': rel.end_node['start_time'],
                                'end_time': rel.end_node['end_time'],
                                'email': rel.end_node['email'],
                                'direction': rel.end_node['direction']
                                }
            result['rel_type'] = relation_type
            json_list.append(result)
        return json_list

    # 6.?????????????????????1+??????+??????2(??????-??????->??????)
    def findEntityRelation(self, entity1, relation, entity2):
        answer = self.graph.run(
            "match (source)-[rel:" + relation + "]->(target)  where source.name= $name1 and target.name = $name2 " +
            "return rel ", name1=entity1, name2=entity2).data()

        json_list = []
        for an in answer:
            result = {}
            rel = an['rel']
            relation_type = list(rel.types())[0]
            start_name = rel.start_node['name']
            start_intro = rel.start_node['introduction']
            end_name = rel.end_node['name']
            end_intro = rel.end_node['introduction']
            result['source'] = {'name': start_name, 'time': rel.start_node['time'],
                                'type': str(rel.start_node.labels),
                                'intro': start_intro,
                                'title': rel.start_node['title'],
                                'image': rel.start_node['image'] or rel.start_node['imageURL'],
                                'website': rel.start_node['website'] or rel.start_node['homepage'],
                                'birth_year': rel.start_node['birthyear'],
                                'death_year': rel.start_node['deathyear'],
                                'start_time': rel.start_node['start_time'],
                                'end_time': rel.start_node['end_time'],
                                'email': rel.start_node['email'],
                                'direction': rel.start_node['direction']
                                }
            result['target'] = {'name': end_name, 'time': rel.end_node['time'],
                                'type': str(rel.end_node.labels),
                                'intro': end_intro,
                                'title': rel.end_node['title'],
                                'image': rel.end_node['image'] or rel.end_node['imageURL'],
                                'website': rel.end_node['website'] or rel.start_node['homepage'],
                                'birth_year': rel.end_node['birthyear'],
                                'death_year': rel.end_node['deathyear'],
                                'start_time': rel.end_node['start_time'],
                                'end_time': rel.end_node['end_time'],
                                'email': rel.end_node['email'],
                                'direction': rel.end_node['direction']
                                }
            result['rel_type'] = relation_type
            json_list.append(result)
        return json_list

    # ???.??????
    '''
	0:nt ????????????
    1:nt ????????????
    2:nt ????????????
    3:t ?????????
    4:t ??????
    5:t ??????
    6:nt ??????????????????
    7:nt ????????????
    8:nt ??????????????????
    9:nt ??????????????????
	'''
    # 0:nt ????????????
    def sub_college(self, name):
        answer = self.graph.run(
            "match (n)-[r:`????????????`]->(target) where n.name = $name return target.name as name", name=name
        ).data()
        answer_dict = self.store_answer(answer)
        return answer_dict

    # 1:nt ????????????
    def trans_in(self, name):
        answer = self.graph.run(
            "match (n)-[r:`??????`]->(target) where target.name=$name return n.name as name", name=name
        ).data()
        answer_dict = self.store_answer(answer)
        return answer_dict

    # 2:nt ????????????
    def trans_out(self, name):
        answer = self.graph.run(
            "match (n)-[r:`??????`]->(target) where n.name=$name return target.name as name", name=name
        ).data()
        answer_dict = self.store_answer(answer)
        return answer_dict

    # 3:t ?????????
    def leader(self, time):
        answer = self.graph.run(
            "match (n:headmaster) where date(n.start_time)<=date($time)<=date(n.end_time) " +
            "return n.name as name", time=time
        ).data()
        answer_dict = self.store_answer(answer)
        return answer_dict

    # 4:t ??????
    def school_name(self, time):
        answer = self.graph.run(
            "match (n:school) where date(n.start_time)<=date($time)<=date(n.end_time) return n.name as name",
            time=time
        ).data()
        answer_dict = self.store_answer(answer)
        return answer_dict

    # 5:t ??????
    def event_name(self, time):
        answer = self.graph.run(
            "match (n:event) where date(n.time)=date($time) return n.name as name",
            time=time
        ).data()
        answer_dict = self.store_answer(answer)
        return answer_dict

    # 6:nt ??????????????????
    def teachers(self, name):
        answer = self.graph.run(
            "match (n)-[r:`?????????`]->(target) where target.name = $name return n.name as name", name=name
        ).data()
        answer_dict = self.store_answer(answer)
        return answer_dict

    # 7:nt ????????????
    def precursor(self, name):
        answer = self.graph.run(
            "match (n)-[r:`??????`]->(target) where n.name = $name return target.name as name", name=name
        ).data()
        answer_dict = self.store_answer(answer)
        return answer_dict

    # 8:nt ??????????????????
    def start_time(self, name):
        answer = self.graph.run(
            "match (n:school) where n.name = $name return n.start_time as name", name=name
        ).data()
        answer_dict = self.store_answer(answer)
        return answer_dict

    # 9:nt ??????????????????
    def teacher_cnt(self, name):
        answer = self.graph.run(
            "match (n)-[r:`?????????`]->(target) where target.name = $name return count(n) as name", name=name
        ).data()
        answer_dict = self.store_answer(answer)
        return answer_dict

    def store_answer(self, answer):
        answer_dict = {}
        answer_name = []
        answer_list = []
        for an in answer:
            result = {'source': {'name': an["name"]}}
            answer_list.append(result)
            answer_name.append(an["name"])
            print(an)
        answer_dict['answer'] = answer_name
        answer_dict['list'] = answer_list
        return answer_dict
