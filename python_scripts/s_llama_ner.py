#for skillspan
import os
import random
import torch
import json
import numpy as np
from datasets import Dataset
from seqeval.metrics import classification_report, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, Trainer, TrainingArguments, AutoConfig
from sklearn.metrics import classification_report
import torch.nn as nn


torch.cuda.empty_cache()


def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return Dataset.from_dict({
        "idx": [example["idx"] for example in data],
        "tokens": [example["tokens"] for example in data],
        "tags_skill": [example["tags_skill"] for example in data]
    })




def count_bio_labels_before(dataset):
    bio_counts = {"B": 0, "I": 0, "O": 0}
    for example in dataset["tags_skill"]:
        for label in example:
            if label in bio_counts:
                bio_counts[label] += 1
    return bio_counts



root_dir = "/mnt/scratch/users/genir/arran"
dataset_dir = root_dir + "/datasets"
dataset_name = "skillspan_dataset"

model_dir = root_dir + "/models"
model_name = "llama2-7b"

preprocessed_dir = root_dir + "/preprocess"
output_dir = root_dir + "/outputs"




train_dataset = load_dataset(os.path.join(dataset_dir, dataset_name, "train_new.json"))
dev_dataset = load_dataset(os.path.join(dataset_dir, dataset_name, "dev_new.json"))
test_dataset = load_dataset(os.path.join(dataset_dir, dataset_name, "test_new.json"))




bio_label_count = {
    "train": count_bio_labels_before(train_dataset),
    "dev": count_bio_labels_before(dev_dataset),
    "test": count_bio_labels_before(test_dataset)
}



print("BIO label counts before:", bio_label_count)
print("\n")


os.makedirs(os.path.join(preprocessed_dir, dataset_name), exist_ok=True)

with open(os.path.join(preprocessed_dir, dataset_name, "bio_count_before_tokenization.json"), 'w') as f:
    json.dump(bio_label_count, f, indent=4)



label_map = {"O": 0, "B": 1, "I": 2}
label_map_inv = {0: "O", 1: "B", 2: "I"}




tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, model_name))


tokenizer.add_special_tokens({"pad_token": "<pad>"})
model.config.pad_token_id = tokenizer.pad_token_id


model_config = AutoConfig.from_pretrained(os.path.join(model_dir, model_name))


model = AutoModelForTokenClassification.from_pretrained(os.path.join(model_dir, model_name), num_labels=3, ignore_mismatched_sizes=True)


model.resize_token_embeddings(len(tokenizer))


model.classifier = torch.nn.Linear(model.config.hidden_size, 3)



def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        padding=True,
        truncation=True,
        max_length=128
    )


    aligned_labels = []
    for i, word_labels in enumerate(examples["tags_skill"]):
        word_ids = tokenized_inputs.word_ids(i)
        aligned_example = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                aligned_example.append(-100)
            elif word_idx != previous_word_idx:
                aligned_example.append(label_map[word_labels[word_idx]])
            else:
                aligned_example.append(label_map["I"] if word_labels[word_idx] != "O" else label_map["O"])

            previous_word_idx = word_idx

        aligned_labels.append(aligned_example)

    return {
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
        "labels": aligned_labels
    }





train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
dev_dataset = dev_dataset.map(tokenize_and_align_labels, batched=True)
test_dataset = test_dataset.map(tokenize_and_align_labels, batched=True)



data_collator = DataCollatorForTokenClassification(tokenizer)



def count_bio_labels_after(dataset, label_map_inv):
    bio_counts = {"B": 0, "I": 0, "O": 0}
    for example in dataset["labels"]:
        for label in example:
            if label != -100:
                label_str = label_map_inv[label]
                if label_str in bio_counts:
                    bio_counts[label_str] += 1
    return bio_counts




bio_label_count = {
    "train": count_bio_labels_after(train_dataset, label_map_inv),
    "dev": count_bio_labels_after(dev_dataset, label_map_inv),
    "test": count_bio_labels_after(test_dataset, label_map_inv)
}


with open(os.path.join(preprocessed_dir, dataset_name, "bio_count_after_tokenization.json"), 'w') as f:
    json.dump(bio_label_count, f, indent=4)


print("BIO label counts after:", bio_label_count)
print("\n")



if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

model = model.cuda()



training_args = TrainingArguments(
    output_dir=os.path.join(output_dir, "skillspan_dataset/results"),
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    #batch size was 16:
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    #logging_dir="./logs",
    logging_steps=1,
    save_steps=500,
    save_total_limit=2,
    no_cuda=False,
    logging_strategy="epoch",

    #after evaluating each epochs performance
    load_best_model_at_end=True,
    #metric_for_best_model="f1",
    ###

    report_to="none",
    fp16=True,
    gradient_accumulation_steps=4,
    remove_unused_columns=False
)

epoch_f1_scores = {"B": [], "I": [], "O": []}

def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(axis=-1)

    mask = labels != -100
    predictions = predictions[mask]
    labels = labels[mask]

    predictions = [label_map_inv[label] for label in predictions.tolist()]
    labels = [label_map_inv[label] for label in labels.tolist()]

    report = classification_report(labels, predictions, target_names=["B", "I", "O"], output_dict=True)



    print(classification_report(labels, predictions, target_names=["B", "I", "O"]))

    epoch_f1_scores["B"].append(report["B"]["f1-score"])
    epoch_f1_scores["I"].append(report["I"]["f1-score"])
    epoch_f1_scores["O"].append(report["O"]["f1-score"])


    with open("/users/40624421/sharedscratch/datasets/skillspan_dataset/s_bert_ner_f1.json", 'w') as f:
        json.dump(epoch_f1_scores, f)

    return {
}



trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    tokenizer=tokenizer
)




print("\n----Begin Training----\n")
trainer.train()
print("\n")



def evaluate_one_shot_example(dev_dataset):

    one_shot_example = dev_dataset[21]


    inputs = tokenizer(one_shot_example["tokens"], is_split_into_words=True, padding=True, truncation=True, return_tensors="pt")
    labels = one_shot_example["tags_skill"]

    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=-1).squeeze().cpu().numpy()

    pred_labels = [label_map_inv[pred] for pred in predictions]
    true_labels = labels

    true_labels = true_labels[:len(pred_labels)]
    pred_labels = pred_labels[:len(true_labels)]


    print("\n")
    print("\n----One-Shot Example on Developement Set----\n")
    print("Example Tokens:", one_shot_example["tokens"])
    print("True Labels:", true_labels)
    print("Predicted Labels:", pred_labels)
    print("\n")

    report = classification_report(true_labels, pred_labels, target_names=["B", "I", "O"], labels=["B", "I", "O"])
    print("\n----Classification Report for 1-Shot Example on Developement Set----\n", report)
    print("\n")
    print("\n")




evaluate_one_shot_example(dev_dataset)



print("\n----Developement Evaluation----\n")

dev_results = trainer.evaluate(dev_dataset)

print("Developement Set Evaluation Results:", dev_results)
print("\n")




print("\n----Test Evaluation----\n")

test_results = trainer.evaluate(test_dataset)

print("Test set results:", test_results)


