import json
import os
import google.generativeai as genai
import argparse
import time
import pickle
import random

query_dict_tail = {
    "en": "Predict the tail entity for triplet ({}, {}, ?). Please give the 10 most possible answers. You just need to give the names of the entities seperated by commas and please do not give explanation."
}

query_dict_head = {
    "en": "Predict the head entity for triplet (?, {}, {}). Please give the 10 most possible answers. You just need to give the names of the entities seperated by commas and please do not give explanation."
}


explain_dict = {
    "en": "In the following task, entities and relations will be expressed with their names. Please notice that some names are replaced with metasyntactic words. You will first be given background knowledge in the form of triplet (h, r, t) which means entity 'h' has relation 'r' with entity 't'. Then you will be asked to predict the missing entity for a triplet. "
}

metasyntactic = ["foobar", "foo", "bar", "baz", "qux", "quux", "corge", "grault", "garply", "waldo", "fred", "plugh", "xyzzy", "thud", "bazola", "ztesch", "grunt", "bletch", "fum", "toto", "titi", "tata", "tutu", "pippo", "pluto", "paperino"]
numMeta = len(metasyntactic)

entity_common = "./entity.pkl"
relation_common = "./relation.pkl"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemini Test")
    parser.add_argument(
        "--language", type=str, default="en"
    )
    parser.add_argument(
        "--dataset", type=str, default="s"
    )
    parser.add_argument(
        "--reverse", type=bool, default=False
    )
    parser.add_argument(
        "--start", type=int, default=0
    )
    parser.add_argument(
        "--end", type=int, default=30
    )
    parser.add_argument(
        "--type", type=str, default="t"
    )
    args = parser.parse_args()
    language = args.language
    dataset = args.dataset

    bg_reverse = args.reverse
    relation_reverse = False
    query_type = args.type

    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')

    entity2id = {}
    relation2id = {}

    with open(entity_common, 'rb') as f:
        commonEntities = pickle.load(f)
    
    with open(relation_common, 'rb') as f:
        commonRelations = pickle.load(f)

    entity_file = "./codex/data/entities/{}/entities.json".format(language)
    relation_file = "./codex/data/relations/{}/relations.json".format(language)
    with open(entity_file, "r", encoding = "utf-8") as f:
        entity_dict = json.load(f)
    with open(relation_file, "r", encoding = "utf-8") as f:
        relation_dict = json.load(f)

    relations = "|"
    relation_list = []
    for relation in relation_dict:
        if relation in commonRelations:
            relation_list.append(relation_dict[relation]['label'])

    if relation_reverse:
        relation_list = relation_list[::-1]

    for relation in relation_list:
        relations += relation + "|"

    entities = "|"
    entity_list = []
    for entity in entity_dict:
        if entity in commonEntities:
            entity_list.append(entity_dict[entity]['label'])

    for entity in entity_list:
        entities += entity + "|"

    bg = "Background knowledge: "
    train_file = "./codex/data/triples/codex-{}/train.txt".format(dataset)
    with open(train_file, "r", encoding = "utf-8") as f:
        train_lines = f.readlines()

    if bg_reverse:
        train_lines = train_lines[::-1]

    for line in train_lines:
        h = line.split()[0].strip()
        r = line.split()[1].strip()
        t = line.split()[2].strip()

        if h in commonEntities and t in commonEntities and r in commonRelations:
            bg += "({}, {}, {}); ".format(entity_dict[h]['label'], relation_dict[r]['label'], entity_dict[t]['label'])

    kg_info = bg[:-2] + ". "

    explain = explain_dict[language]

    test_file = "./codex/data/triples/codex-{}/test.txt".format(dataset)
    with open(test_file, "r", encoding = "utf-8") as f:
        lines = f.readlines()

    count = 0
    total = 0
    start = args.start
    end = args.end
        
    lines = lines[start:end]    
    count_num = start

    for line in lines:
        random.shuffle(metasyntactic)

        count_num += 1
        h = line.split()[0].strip()
        r = line.split()[1].strip()
        t = line.split()[2].strip()

        if h in commonEntities and t in commonEntities and r in commonRelations:
            total += 1

            if query_type == 't':
                query = query_dict_tail[language].format(entity_dict[h]['label'], relation_dict[r]['label'])
            else:
                query = query_dict_head[language].format(relation_dict[r]['label'], entity_dict[t]['label'])

            prompt = explain + kg_info + query
           
            prompt = prompt.replace(entity_dict[h]["label"], metasyntactic[0])
            prompt = prompt.replace(entity_dict[t]["label"], metasyntactic[1])
            prompt = prompt.replace(relation_dict[r]["label"], metasyntactic[2])

            replace = [h, r, t]
            replace_dict = {metasyntactic[0]:entity_dict[h]["label"], metasyntactic[1]:entity_dict[t]["label"], metasyntactic[2]: relation_dict[r]["label"]}
            i = 3

            if query_type == 't':
                target = h
                label = metasyntactic[1]
            else:
                target = t
                label = metasyntactic[0]

            for train_line in train_lines:
                train_h = train_line.split()[0].strip()
                train_r = train_line.split()[1].strip()
                train_t = train_line.split()[2].strip()
                if train_h == target:
                    if i < numMeta:
                        if train_r not in replace:
                            replace.append(train_r)
                            replace_dict[metasyntactic[i]] = relation_dict[train_r]["label"]
                            prompt = prompt.replace(relation_dict[train_r]["label"], metasyntactic[i])
                            i += 1
                    if i < numMeta:
                        if train_t not in replace:
                            replace.append(train_t)
                            replace_dict[metasyntactic[i]] = entity_dict[train_t]["label"]
                            prompt = prompt.replace(entity_dict[train_t]["label"], metasyntactic[i])
                            i += 1
                    if i == numMeta:
                        break
                if train_t == target:
                    if i < numMeta:
                        if train_r not in replace:
                            replace.append(train_r)
                            replace_dict[metasyntactic[i]] = relation_dict[train_r]["label"]
                            prompt = prompt.replace(relation_dict[train_r]["label"], metasyntactic[i])
                            i += 1
                    if i < numMeta:
                        if train_h not in replace:
                            replace.append(train_h)
                            replace_dict[metasyntactic[i]] = entity_dict[train_h]["label"]
                            prompt = prompt.replace(entity_dict[train_h]["label"], metasyntactic[i])
                            i += 1
                    if i == numMeta:
                        break

            response = model.generate_content(prompt)

            if label in response.text:
                count += 1

            print("query: " + query)
            print("ground truth: " + label)
            print("answer: " + response.text)

            time.sleep(15)

    print("codex-{} count: {}, total: {}, hits@10: {}".format(dataset, count, total, float(count/total)))