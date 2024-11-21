import json
import os
import google.generativeai as genai
import argparse
import time
import pickle
import random

query_dict = {
    "en": "What is the relationship between entity '{}' and entity '{}'? Please choose one best answer from the following relations:{}. You just need to give the answer and please do not give explanation."
}

explain_dict = {
    "en": "In the following task, you will first be given background knowledge in the form of triplet (h, r, t) which means entity 'h' has relation 'r' with entity 't'. Then you will be asked some questions about the relationship between entities. Please notice that some words are replaced with metasyntactic words in the following paragraph. "
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
    args = parser.parse_args()
    language = args.language
    dataset = args.dataset

    bg_reverse = args.reverse
    relation_reverse = False

    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-pro-latest')

    entity2id = {}
    relation2id = {}

    with open(entity_common, 'rb') as f:
        commonEntities = pickle.load(f)
    
    entityID = [i for i in range(len(commonEntities))]
    random.shuffle(entityID)
    i = 0
    for entity in commonEntities:
        entity2id[entity] = str(entityID[i])
        i += 1
    
    with open(relation_common, 'rb') as f:
        commonRelations = pickle.load(f)

    relationID = [i for i in range(len(commonRelations))]
    random.shuffle(relationID)
    i = 0
    for relation in commonRelations:
        relation2id[relation] = str(relationID[i])
        i += 1

    entity_file = "./codex/data/entities/{}/entities.json".format(language)
    relation_file = "./codex/data/relations/{}/relations.json".format(language)
    with open(entity_file, "r", encoding = "utf-8") as f:
        entity_dict = json.load(f)
    with open(relation_file, "r", encoding = "utf-8") as f:
        relation_dict = json.load(f)

    entity_mapping = "Entity mapping: "
    for entity in entity2id:
        entity_mapping += "{} is entity '{}'; ".format(entity_dict[entity]['label'], entity2id[entity])
    entity_mapping = entity_mapping[:-2] + '. '

    relation_mapping = "Relation mapping: "
    for relation in relation2id:
        relation_mapping += "{} is relation '{}'; ".format(relation_dict[relation]['label'], relation2id[relation])
    relation_mapping = relation_mapping[:-2] + '. '

    relations = "|"
    relation_list = []
    for relation in relation_dict:
        if relation in commonRelations:
            relation_list.append(relation)

    if relation_reverse:
        relation_list = relation_list[::-1]

    for relation in relation_list:
        relations += relation_dict[relation]["label"] + "|"

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
            bg += "({}, {}, {}); ".format(entity_dict[h]["label"], relation_dict[r]["label"], entity_dict[t]["label"])

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

    for line in lines:
        random.shuffle(metasyntactic)

        h = line.split()[0].strip()
        r = line.split()[1].strip()
        t = line.split()[2].strip()

        if h in commonEntities and t in commonEntities and r in commonRelations:
            label = metasyntactic[2]

            total += 1
            query = query_dict[language].format(entity_dict[h]["label"], entity_dict[t]["label"], relations)

            prompt = explain + kg_info + query

            prompt = prompt.replace(entity_dict[h]["label"], metasyntactic[0])
            prompt = prompt.replace(entity_dict[t]["label"], metasyntactic[1])
            prompt = prompt.replace(relation_dict[r]["label"], metasyntactic[2])

            replace = [h, r, t]
            replace_dict = {metasyntactic[0]:entity_dict[h]["label"], metasyntactic[1]:entity_dict[t]["label"], metasyntactic[2]: relation_dict[r]["label"]}
            i = 3
            for train_line in train_lines:
                train_h = train_line.split()[0].strip()
                train_r = train_line.split()[1].strip()
                train_t = train_line.split()[2].strip()
                if train_h == h:
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
                if train_t == h:
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
            print("ground truth: " + relation_dict[r]["label"])
            if response.text in replace_dict:
                print("answer: " + replace_dict[response.text])
            else:
                print("answer: " + response.text)

            time.sleep(10)

    print("codex-{} count: {}, total: {}, hits@1: {}".format(dataset, count, total, float(count/total)))