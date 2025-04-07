import torch
import json
import os
import numpy as np
import wandb
from datasets import Dataset
from seqeval.metrics import classification_report, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, Trainer, TrainingArguments, AutoConfig
from sklearn.metrics import classification_report, accuracy_score



#function to load in the dataset. The 'idx' can be commented out when loading the LinkedIn Dataset
def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return Dataset.from_dict({
        "idx": [example["idx"] for example in data],
        "tokens": [example["tokens"] for example in data],
        "tags_skill": [example["tags_skill"] for example in data]
    })



root_dir = "/mnt/scratch/users/genir/arran"
dataset_dir = root_dir + "/datasets"
dataset_name = "skillspan_dataset"

model_dir = root_dir + "/models"
model_name = "bert-base-NER"

preprocessed_dir = root_dir + "/preprocess/bert-base-NER"
output_dir = root_dir + "/outputs/bert-base-NER"


train_dataset = load_dataset(os.path.join(dataset_dir, dataset_name, "train_new.json"))
dev_dataset = load_dataset(os.path.join(dataset_dir, dataset_name, "dev_new.json"))
test_dataset = load_dataset(os.path.join(dataset_dir, dataset_name, "test_new.json"))



os.makedirs(os.path.join(preprocessed_dir, dataset_name), exist_ok=True)


label_map = {"O": 0, "B": 1, "I": 2}
label_map_inv = {0: "O", 1: "B", 2: "I"}



#this code is from the huggingface page on how to load the model
tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, model_name))

model_config = AutoConfig.from_pretrained(os.path.join(model_dir, model_name))

#had to sepcify the number of labels is 3 for our classes B, I and O. This model was originally using 9 labels
model = AutoModelForTokenClassification.from_pretrained(os.path.join(model_dir, model_name), num_labels=3, ignore_mismatched_sizes=True)

model.classifier = torch.nn.Linear(model.config.hidden_size, 3)




#code for the bert tokenizer. This also adds padding and creates the attention mask.
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



#applyign the tokenizer to our three subsets
train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
dev_dataset = dev_dataset.map(tokenize_and_align_labels, batched=True)
test_dataset = test_dataset.map(tokenize_and_align_labels, batched=True)

data_collator = DataCollatorForTokenClassification(tokenizer)




#counting the number of B, I and O after the tokenizer added padding to the original dataset. This information is displayed
#in a graph in my dissertation.
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


print("BIO label counts after tokenization:", bio_label_count)
print("\n")




#specifying the metrics for each class in our datasets: B, I and O. Using the import 'classification report' to
#display the precision, recall and F1 score for each of the classes 
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


    return {
        "accuracy": accuracy
    }




print("\n---Hyperparameter Tuning----\n")

#using 'wand' to log our optuna runs. Saving them offline because linking clusetr to wandb online was taking too long
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_DISABLED"] = "true"


def optuna_hp_space(trial):
    print("\n")
    batch_size = trial.suggest_categorical("per_device_train_batch_size", [16, 32, 64, 128])
    #learning_rate = trial.suggest_float("learning_rate", 1e-5, 3e-5)
    learning_rate = trial.suggest_categorical("learning_rate", [1e-5, 2e-5, 3e-5, 4e-5])
    print("\n----Batch Size: " + str(batch_size) + ", Learning Rate: " + str(learning_rate) + "----\n")

    wandb.init(project="s_bert_ner_hp", config={"batch_size": batch_size, "learning_rate": learning_rate})

    return {
        "per_device_train_batch_size": batch_size,
        "learning_rate": learning_rate
    }



def model_init(trial):
    model = AutoModelForTokenClassification.from_pretrained(
        "/mnt/scratch/users/genir/arran/models/bert-base-NER",
        num_labels=3,
        ignore_mismatched_sizes=True
    )
    wandb.finish()

    return model


#defining the trainign arguments for the hyperparameter training
temp_training_args = TrainingArguments(
    output_dir=os.path.join(output_dir, "skillspan_dataset/vit-sweeps"),
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_steps=None,
    num_train_epochs=1,
    save_total_limit=1,
    logging_strategy="epoch",
    load_best_model_at_end=True,
    #report_to="wandb",
    # fp16=True
    # warmup_steps=0,
    lr_scheduler_type='constant'
    # weight_decay=0.01
)


#usign the compute metrics defined earlier
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

#finding the bestperforming trial and saving it to a variable so it can be used later
best_trial = trainer.hyperparameter_search(
    direction="maximize",
    backend="optuna",
    hp_space=optuna_hp_space,
    n_trials=1,
)


#wandb.finish()



#printing the best trial with its accuracy, batchg size and learning rate
print("\n----Best Trial:----\n", best_trial)


#these training arguments are for the main models training process. Using the specific batch szie and learning rate from 'best trial'
training_args = TrainingArguments(
    output_dir=os.path.join(output_dir, "skillspan_dataset/results"),
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=best_trial.hyperparameters["learning_rate"],
    per_device_train_batch_size=best_trial.hyperparameters["per_device_train_batch_size"],
    per_device_eval_batch_size=best_trial.hyperparameters["per_device_train_batch_size"],
    num_train_epochs=1,
    weight_decay=0.01,
    save_steps=None,
    save_total_limit=1,
    logging_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none",
    lr_scheduler_type='constant'
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

#best_model = trainer.model

print("\n----Developement Evaluation----\n")

dev_results = trainer.evaluate(dev_dataset)

print("Developement Set Evaluation Results:", dev_results)
print("\n")




print("\n----Test Evaluation----\n")

test_results = trainer.evaluate(test_dataset)

print("Test set results:", test_results)


