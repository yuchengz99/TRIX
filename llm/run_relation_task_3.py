import json
import os
import google.generativeai as genai
import argparse
import time
import pickle
import random

query_dict = {
    "en": "What is the relationship between entity '{}' and entity '{}'? Please choose one best answer from the following relation IDs:{}. You just need to give the answer and please do not give explanation."
}

explain_dict = {
    "en": "In the following task, entities and relations will be expressed with their IDs. You will first be given the mapping from entities to their IDs and the mapping from relations to their IDs. Then you will be given background knowledge in the form of triplet (h, r, t) which means entity 'h' has relation 'r' with entity 't'. Finally you will be asked some questions about the relationship between entities. "
}

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
        lines = f.readlines()

    if bg_reverse:
        lines = lines[::-1]

    for line in lines:
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
        h = line.split()[0].strip()
        r = line.split()[1].strip()
        t = line.split()[2].strip()

        if h in commonEntities and t in commonEntities and r in commonRelations:
            total += 1
            query = query_dict[language].format(entity_dict[h]["label"], entity_dict[t]["label"], relations)

            prompt = explain + kg_info + query

            response = model.generate_content(prompt)

            if r == response.text:
                count += 1

            print("query: " + query)
            print("ground truth: " + r)
            print("answer: " + response.text)

            time.sleep(15)

    print("codex-{} count: {}, total: {}, hits@1: {}".format(dataset, count, total, float(count/total)))