"""Fine-tune Legal-BERT for NER on mixed LLM and combined open source data."""
## disclaimer: these files were created by the data scientists by asking claude to convert my fine-tuning notebook into a script that can run on their virtual machine

import argparse
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

SEED = 67
MODEL_CHECKPOINT = "models/legal-bert-base-uncased"
BATCH_SIZE = 16
NUM_EPOCHS = 6
LEARNING_RATE = 2e-5
OUTPUT_DIR = Path("legalbert-ner-mixed")
SAVE_DIR = Path("finetuned_legalbert_mixed")
TRAIN_FILE = Path("data/MIXED_TRAIN.conll")
VAL_FILE = Path("data/MIXED_DEV.conll")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output/save directory (e.g. Azure ML output mount)")
    parser.add_argument("--fp16", action="store_true",
                        help="Enable mixed-precision training")
    return parser.parse_args()


def get_list_of_sentences(file_path):
    """this returns a list of lists for sentences and list of lists for labels"""
    current_sent = []
    current_label = []
    all_sents = []
    all_labels = []

    with open(file_path, "r", encoding="utf-8") as infile:
        lines = infile.readlines()
        for line in lines:
            line = line.strip()
            if line == "":
                if len(current_sent) > 0:
                    all_sents.append(current_sent)
                    all_labels.append(current_label)
                    current_sent = []
                    current_label = []
            else:
                splitted = line.split("\t")
                token_part = splitted[0]
                label_part = splitted[1]
                current_sent.append(token_part)
                current_label.append(label_part)

        if len(current_sent) > 0:
            all_sents.append(current_sent)
            all_labels.append(current_label)
    
    return all_sents, all_labels


def main():
    args = parse_args()
    set_seed(SEED)

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    save_dir = output_dir / "finetuned_legalbert_mixed" if args.output_dir else SAVE_DIR

    train_tokens, train_labels = get_list_of_sentences(TRAIN_FILE)
    val_tokens, val_labels = get_list_of_sentences(VAL_FILE)

    train_dataset = Dataset.from_dict({"tokens": train_tokens, "ner_tags": train_labels})
    val_dataset = Dataset.from_dict({"tokens": val_tokens, "ner_tags": val_labels})

    label_list = sorted(set(label for sentence in train_labels for label in sentence))
    label_to_id = {label: i for i, label in enumerate(label_list)}
    id_to_label = {i: label for label, i in label_to_id.items()}

    print(f"Labels ({len(label_list)}): {label_list}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    def tokenize_and_align_labels(examples):
        """
        this tokenizes the sentences and aligns the labels with the tokenized subwords
        """
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
        #tokenized the lists of sentences
        all_labels = []
        for i, word_labels in enumerate(examples["ner_tags"]): #for every label list in the ner_tags we have
            word_ids = tokenized_inputs.word_ids(batch_index=i) #mapping every subword to the original word index
            aligned_labels = [] 
            previous_word_id = None #looking at the previous word id to detect new ones
            for word_id in word_ids: #going through each token
                #special tokens have a None word id, they are set to -100 to be ignored
                #in the loss function
                if word_id is None: #special tokens
                    aligned_labels.append(-100)
                elif word_id != previous_word_id: #if not equal to same as prev, new word!
                    aligned_labels.append(label_to_id[word_labels[word_id]]) #assign it the gold lab of the first subword
                else: #continued subword of the same word
                    label = word_labels[word_id] #getting the OG label
                    if label.startswith("B-"): #check if the label begins a NE
                        aligned_labels.append(label_to_id["I-" + label[2:]]) #changing the rest to I-
                    else:
                        aligned_labels.append(label_to_id[label]) #if label already an I- or O it stays the same
                previous_word_id = word_id #update the prev word for next iteration

            all_labels.append(aligned_labels) #appending list of labels to overall list

        tokenized_inputs["labels"] = all_labels #add the aligned labels to the tokenized input
        return tokenized_inputs

    tokenized_train = train_dataset.map(tokenize_and_align_labels, batched=True).remove_columns(["tokens", "ner_tags"])
    tokenized_val = val_dataset.map(tokenize_and_align_labels, batched=True).remove_columns(["tokens", "ner_tags"])

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=len(label_list),
        id2label=id_to_label,
        label2id=label_to_id,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        seed=SEED,
        fp16=args.fp16,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    save_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))

if __name__ == "__main__":
    main()
