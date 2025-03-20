### Skill extraction from resume using Large language models.


## Datasets

The first dataset is the skillspan dataset which contains text from resumes and job postings. The text is split into sentences. The dataset is partitioned into three subsets: train, development, and test. The training subset contains 4800 sentences, the development subset contains 3174 sentences, and the test subset contains 3569 sentences.

The second dataset is a set of job postings and skills from linkedIn.
This dataset has been transformed to match the format of the Skillspan dataset. This makes comparison easier.


## Models

We used a BERT model from huggingface which was fine-tuned on the English version of the standard CoNLL-2003 Named Entity Recognition dataset.


## Current Experiments

The first experiment uses the BERT model for skill detection with the Skillspan and linkedIn datasets.
We are using hyperparameters to vary the batch size and learning rate.
The code will be run for 16 trials and 40 epochs.


The second experiment will swap the BERT model for another model to see how the performance changes


The third experiment will use data augmentation techniques to create synthetic data.
Specifically, this will be aimed at adding more entries for the underrepresented classes in the existing datasets.


## Progress

Currently, I am trying to fix the issues with the BERT NER hyperparameter code.
Then I can move on to using simple data augmentation techniques with the skillspan and linkedIn datasets.
These techniques include synonym replacement, word deletion and word swap.
