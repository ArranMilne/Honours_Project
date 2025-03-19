#for skillspan
import random
import torch
import json
import os
import shutil
import numpy as np
from datasets import Dataset
from seqeval.metrics import classification_report, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, Trainer, TrainingArguments, AutoConfig
from sklearn.metrics import classification_report, accuracy_score
import wandb


#os.environ["WANDB_PROJECT"] = "vit_snacks_sweeps"
#os.environment["WANDB_LOG_MODEL"] = "true"



def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return Dataset.from_dict({
        "idx": [example["idx"] for example in data],
        "tokens": [example["tokens"] for example in data],
        "tags_skill": [example["tags_skill"] for example in data]
    })



train_dataset = load_dataset("/users/40624421/sharedscratch/datasets/skillspan_dataset/train_new.json")
dev_dataset = load_dataset("/users/40624421/sharedscratch/datasets/skillspan_dataset/dev_new.json")
test_dataset = load_dataset("/users/40624421/sharedscratch/datasets/skillspan_dataset/test_new.json")



def count_bio_labels_before(dataset):
    bio_counts = {"B": 0, "I": 0, "O": 0}
    for example in dataset["tags_skill"]:
        for label in example:
            if label in bio_counts:
                bio_counts[label] += 1
    return bio_counts


bio_label_count = {
    "train": count_bio_labels_before(train_dataset),
    "dev": count_bio_labels_before(dev_dataset),
    "test": count_bio_labels_before(test_dataset)
}


with open("/users/40624421/sharedscratch/datasets/skillspan_dataset/bio_count_before_tokenization.json", 'w') as f:
    json.dump(bio_label_count, f, indent=4)


print("BIO label counts before:", bio_label_count)
print("\n")


#this line is from huggingface page on how to load model
tokenizer = AutoTokenizer.from_pretrained("/users/40624421/sharedscratch/models/bert-base-NER")


label_map = {"O": 0, "B": 1, "I": 2}
label_map_inv = {0: "O", 1: "B", 2: "I"}



model_config = AutoConfig.from_pretrained("/users/40624421/sharedscratch/models/bert-base-NER")


model = AutoModelForTokenClassification.from_pretrained("/users/40624421/sharedscratch/models/bert-base-NER", num_labels=3, ignore_mismatched_sizes=True)


model.classifier = torch.nn.Linear(model.config.hidden_size, 3)



def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        padding=True,
        truncation=True,
        max_length=512
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

with open("/users/40624421/sharedscratch/datasets/skillspan_dataset/bio_count_after_tokenization.json", 'w') as f:
    json.dump(bio_label_count, f, indent=4)


print("BIO label counts after:", bio_label_count)
print("\n")



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

    accuracy = accuracy_score(labels, predictions)

    print(classification_report(labels, predictions, target_names=["B", "I", "O"]))


    epoch_f1_scores["B"].append(report["B"]["f1-score"])
    epoch_f1_scores["I"].append(report["I"]["f1-score"])
    epoch_f1_scores["O"].append(report["O"]["f1-score"])


    with open("/users/40624421/sharedscratch/datasets/skillspan_dataset/s_bert_ner_f1.json", 'w') as f:
        json.dump(epoch_f1_scores, f)

    return {

    "accuracy": accuracy
}



print("\n---Hyperparameter Tuning----\n")

def optuna_hp_space(trial):
    print("\n")
    batch_size = trial.suggest_categorical("per_device_train_batch_size", [16, 32, 64, 128])
    learning_rate = trial.suggest_categorical("learning_rate", [1e-5, 2e-5, 3e-5, 5e-5])
    print("\n----Batch Size: " + str(batch_size) + ", Learning Rate: " + str(learning_rate) + "----\n")

    return {
        "per_device_train_batch_size": batch_size,
	"learning_rate": learning_rate
    }



def model_init(trial):
    return AutoModelForTokenClassification.from_pretrained(
        "/users/40624421/sharedscratch/models/bert-base-NER",
        num_labels=3,
        ignore_mismatched_sizes=True
    )



#os.environ["WANDB_MODE"] = "offline"

sweep_config = {
    "method": "random"
}


parameters_dict = {
    "epochs": {
        "value": 2
        },
    "batch_size": {
        "values": [16, 32, 64, 128]
        },
    "learning_rate": {
        "values": [1e-5, 2e-5, 3e-5, 5e-5]
    },
    #"weight_decay": {
     #   "values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    #},
}


sweep_config["parameters"] = parameters_dict

sweep_id = wandb.sweep(sweep_config, project="s_bert_ner_hp")


def train(config=None):
  with wandb.init(config=config):
    config = wandb.config


temp_training_args = TrainingArguments(
    output_dir="/users/40624421/sharedscratch/datasets/skillspan_dataset/with_hp/temp/vit-sweeps",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_steps=None,
    num_train_epochs=2,
    save_total_limit=1,
    logging_strategy="epoch",
    load_best_model_at_end=True,
    report_to="wandb",
)


trainer = Trainer(
    model=None,
    args=temp_training_args,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    tokenizer=tokenizer,
    model_init=model_init
)

best_trial = trainer.hyperparameter_search(
    direction="maximize",
    backend="optuna",
    hp_space=optuna_hp_space,
    n_trials=2,
)


wandb.agent(sweep_id, train, count=20)

print("\n----Best Trial:----\n", best_trial)
runs_directory = "/users/40624421/sharedscratch/datasets/skillspan_dataset/with_hp/temp/results"
all_runs = [d for d in os.listdir(runs_directory) if os.path.isdir(os.path.join(runs_directory, d))]



training_args = TrainingArguments(
    output_dir="/users/40624421/sharedscratch/datasets/skillspan_dataset/with_hp/results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=best_trial.hyperparameters["learning_rate"],
    per_device_train_batch_size=best_trial.hyperparameters["per_device_train_batch_size"],
    per_device_eval_batch_size=best_trial.hyperparameters["per_device_train_batch_size"],
    num_train_epochs=2,
    weight_decay=0.01,
    save_steps=None,
    save_total_limit=1,
    logging_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none",
)


trainer = Trainer(
    model=None,
    args=training_args,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    tokenizer=tokenizer,
    model_init=model_init
)


print("\n----Using Batch Size: " + str(training_args.per_device_train_batch_size) + " with Learning Rate: " + str(training_args.learning_rate) + "----\n")

print("\n----Begin Training----\n")
trainer.train()
print("\n")


best_model = trainer.model


def evaluate_one_shot_example(dev_dataset, model):

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
    print("Example Tokens:", one_shot_example['tokens'])
    print("True Labels:", true_labels)
    print("Predicted Labels:", pred_labels)
    print("\n")

    report = classification_report(true_labels, pred_labels, target_names=["B", "I", "O"], labels=["B", "I", "O"])
    print("\n----Classification Report for 1-Shot Example on Developement Set----\n", report)
    print("\n")
    print("\n")




evaluate_one_shot_example(dev_dataset, best_model)


print("\n----Developement Evaluation----\n")

dev_results = trainer.evaluate(dev_dataset)

print("Developement Set Evaluation Results:", dev_results)
print("\n")



print("\n----Test Evaluation----\n")

test_results = trainer.evaluate(test_dataset)

print("Test set results:", test_results)
