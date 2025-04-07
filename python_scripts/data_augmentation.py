import json
import re
import gensim




#da to find a synonym for each B and I skill in the dataset, and add it next to the original. This increases the number of skills in the dataset.
#Using word2vecs library to find word synonyms. This was more accurate than when I tried to use 'wordnet'.
word2vec_sample = '/users/40624421/sharedscratch/models/models/word2vec_sample/pruned.word2vec.txt'

#using gensim is easiest way to load word2vec
model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_sample, binary=False)

#using our original linkedIn train file after tokenisation and BIO tagging has been applied
original_train_file = "/users/40624421/sharedscratch/datasets/skillspan_dataset/train_new.json"

with open(original_train_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

for job in data:

    #the exisitng lists of tokens and skill tags for each job in the dataset
    tokens = job["tokens"]
    tags_skill = job["tags_skill"]

    #lists to store the new synonyms tokens and tags
    new_tokens = []
    new_tags_skill = []

    #going through each word and its skill tag in each job text
    for token, tag in zip(tokens, tags_skill):
        new_tokens.append(token)
        new_tags_skill.append(tag)

	#if the tag of a word is B or I this means that the word is a skill and a synonym should be added for it
        if tag == "B" or tag == "I":
            #need exception handling because an error will occur when word2vec cannot find a synonym for a skill.
            #these are skipped and no synonym is added.
            try:
                similar_words = model.most_similar(positive=[token], topn=5)
                #using the fourth mnost common word because the first three are usually only different tenses of the same word.
                #using these is not adding additional skills. Adding the synonym as a new token in each jobs list of tokens.
                new_tokens.append(similar_words[4][0])
                #saving the new tag to a list to be added to the existing list
                new_tags_skill.append(tag)
            except KeyError:

                continue

    #updating the original columns to contain the new synonyms and new skill tags for those synonyms.
    job["tokens"] = new_tokens
    job["tags_skill"] = new_tags_skill

with open("/users/40624421/sharedscratch/datasets/skillspan_dataset/skillspan_train_word2vec.json", "w") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)


sys.exit()







