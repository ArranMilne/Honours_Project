import pandas as pd
import json
import re
import random
import nltk
from nltk.tokenize import word_tokenize



#variable to set the maximum number of words that a sentence in the dataset should be. This helps
#to reduce some of the extremely long sentences which would make the dataset unbalanced.
max_words = 25

#regex patterns defined. First splits on punctuation to make sentences. Second one splits the sentences into individual words.
#Keeps each words apostraphies so that they can still be matched easily with the skill list.
rgx_1 = r"'(?<=[.!?;:])\s*"
rgx_2 = r"\b\w+(?:'\w+)?\b|[^\w\s]"



#job_summary = "/users/40624421/sharedscratch/datasets/linkedIn_dataset/job_summary.json"
#job_skills = "/users/40624421/sharedscratch/datasets/linkedIn_dataset/job_skills.json"

#df_summary = pd.read_json(job_summary)
#df_skills = pd.read_json(job_skills)


#using the 'job_link' to match job summaries with job skills
#df_merged = pd.merge(df_summary, df_skills, on="job_link", how="inner")
#data = df_merged.to_dict(orient="records")


#used for testing the code with a smaller file
input_file = "/users/40624421/sharedscratch/datasets/linkedIn_dataset/test.json"
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)



#first change to the dataset is to add the additional skills from the jobs 'skill list' to
#the job text to create the relaxed version of the dataset. This has to be done before sentence splitting
#so that each job has all the text it requires.

for job in data:

    #lowercasing each skill in the list of skills and the job text so that there are no case issues when
    #matching skills in the BIO tagging section.
    skills = [skill.lower() for skill in job["job_skills"]]
    job_summary = job["job_summary"].lower()

    #finding the skills which do not already appear in the jobs skill list
    missing_skills = [skill for skill in skills if skill not in job_summary]

    #the missing skills are added at the end of the job summary text with a set message
    if missing_skills:
        job_summary += ". Additional skills required: " + ", ".join(missing_skills)

    #updating the job summary column to contain the new skills
    job["job_summary"] = job_summary

    #splitting the job text into senetcnes by splitting on punctuation.
    sentences = re.findall(rgx_1, job_summary.strip())

    #list to store the new sentences
    modified_sentences = []
    
    #going through each sentence in one jobs summary text
    for sentence in sentences:

        words = sentence.split()
        #using the variable 'max_words' to make sure sentences cannot be longer than 25 words.
	#this helps to balance the dataset
        for i in range(0, len(words), max_words):
            modified_sentences.append(" ".join(words[i:i + max_words]))

    #adding a new column to the dataset to display the job summaries in sentences
    job["sentences"] = modified_sentences


    #removing these columns as they are no longer needed. job summary is replaced with sentences
    job.pop("job_summary", None)
    job.pop("job_link", None)




#setting the list to store the new set of words and tags for each sentence. This is added to at the end of the loop.
structured_data = []


for job in data:

    job_skills = job["job_skills"]
    sentences_list = job["sentences"]

    #going through each sentence in the current job
    for sentence in sentences_list:
        #tokenizing each sentence, using the regex variable defined earlier.
        sentence_tokens = re.findall(rgx_2, sentence)
        if not sentence_tokens:
            continue
        #creating a list of tags which matches the length of tokens.
        tags = ["O"] * len(sentence_tokens)
        #looping through each skill in the jobs 'job_skills' list
        for skill in job_skills:
            #tokenizing each skill in the current jobs skill list
            skill_tokens = re.findall(rgx_2, skill)

            skill_len = len(skill_tokens)

            #skipping if the skill is longer than the tokens in the sentence. This saves time.
            if skill_len > len(sentence_tokens):
                continue

            #going through every tokenized word in the sentence
            for i in range(len(sentence_tokens) - skill_len + 1):

                #checking if the current segment of tokens matches any of the skill tokens. This means a skill
		#has been found in the job summary
                sentence_chunk = [word.lower() for word in sentence_tokens[i:i + skill_len]]
                skill_chunk = [word.lower() for word in skill_tokens]

                if sentence_chunk == skill_chunk:

                    #if a match is found, the first word is given the tag 'B'
                    if i < len(tags):
                        tags[i] = "B"
                    for j in range(1, skill_len):
                        if i + j < len(tags):
                            tags[i + j] = "I"
        structured_data.append({"tokens": sentence_tokens, "tags_skill": tags})




#these sizes are set to match the skillspan subsets
#train_size = 4800
#dev_size = 3174
#test_size = 3569

#testing
train_size = 20

train_data = structured_data[:train_size]
#dev_data = structured_data[train_size:train_size + dev_size]
#test_data = structured_data[train_size + dev_size:train_size + dev_size + test_size]


#saving the train, dev and test subsets to three seperate files so that theey can each be loaded individually when we work with BERT and Llama2-7b
with open("/users/40624421/sharedscratch/datasets/linkedIn_dataset/linkedIn_train_test.json", "w") as f:
    json.dump(train_data, f, indent=4, ensure_ascii=False)

#with open("/users/40624421/sharedscratch/datasets/linkedIn_dataset/linkedIn_dev.json", "w") as f:
#    json.dump(dev_data, f, indent=4, ensure_ascii=False)

#with open("/users/40624421/sharedscratch/datasets/linkedIn_dataset/linkedIn_test.json", "w") as f:
#    json.dump(test_data, f, indent=4, ensure_ascii=False)











