"""Fine-tune RoBERTa for NER on LLM generated CoNLL data."""
## disclaimer: these files were created by the data scientists by asking claude to convert my fine-tuning notebook into a script that can run on their virtual machine

import argparse
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from seqeval.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    RobertaTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)


SEED = 67
MODEL_CHECKPOINT = "models/roberta-base"
BATCH_SIZE = 16
NUM_EPOCHS = 6
LEARNING_RATE = 2e-5
OUTPUT_DIR = Path("roberta-ner")
SAVE_DIR = Path("finetuned_roberta_LLM")
TRAIN_FILE = Path("data/LLM_train.conll")
VAL_FILE = Path("data/LLM_dev.conll")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output/save directory (e.g. Azure ML output mount)")
    parser.add_argument("--fp16", action="store_true",
                        help="Enable mixed-precision training")
    return parser.parse_args()



def read_conll_to_lists(filepath):
    all_tokens = []
    all_labels = []
    current_tokens = []
    current_labels = []

    with open(filepath, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.rstrip("\n")
            if line.strip() == "":
                if current_tokens:
                    all_tokens.append(current_tokens)
                    all_labels.append(current_labels)
                    current_tokens = []
                    current_labels = []
            else:
                parts = line.split("\t")
                if len(parts) == 2:
                    token, label = parts[0], parts[1]
                    current_tokens.append(token)
                    current_labels.append(label)
        if current_tokens:
            all_tokens.append(current_tokens)
            all_labels.append(current_labels)

    return all_tokens, all_labels



def main():
    args = parse_args()
    set_seed(SEED)

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    save_dir = output_dir / "finetuned_roberta_LLM" if args.output_dir else SAVE_DIR

    print("Loading data...")
    train_tokens, train_labels = read_conll_to_lists(TRAIN_FILE)
    val_tokens, val_labels = read_conll_to_lists(VAL_FILE)

    train_dataset = Dataset.from_dict({"tokens": train_tokens, "ner_tags": train_labels})
    val_dataset = Dataset.from_dict({"tokens": val_tokens, "ner_tags": val_labels})

    # Build label mappings
    label_list = sorted(set(label for sentence in train_labels for label in sentence))
    label_to_id = {label: i for i, label in enumerate(label_list)}
    id_to_label = {i: label for label, i in label_to_id.items()}

    print(f"Labels ({len(label_list)}): {label_list}")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [id_to_label[p] for p, l in zip(pred, lab) if l != -100]
            for pred, lab in zip(predictions, labels)
        ]
        true_labels = [
            [id_to_label[l] for p, l in zip(pred, lab) if l != -100]
            for pred, lab in zip(predictions, labels)
        ]

        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
        }
    # Tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_CHECKPOINT, add_prefix_space=True)

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
        )

        all_labels = []
        for i, word_labels in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            aligned_labels = []
            previous_word_id = None

            for word_id in word_ids:
                if word_id is None:
                    aligned_labels.append(-100)
                elif word_id != previous_word_id:
                    aligned_labels.append(label_to_id[word_labels[word_id]])
                else:
                    label = word_labels[word_id]
                    if label.startswith("B-"):
                        aligned_labels.append(label_to_id["I-" + label[2:]])
                    else:
                        aligned_labels.append(label_to_id[label])

                previous_word_id = word_id

            all_labels.append(aligned_labels)

        tokenized_inputs["labels"] = all_labels
        return tokenized_inputs

    print("Tokenizing...")
    tokenized_train = train_dataset.map(tokenize_and_align_labels, batched=True).remove_columns(["tokens", "ner_tags"])
    tokenized_val = val_dataset.map(tokenize_and_align_labels, batched=True).remove_columns(["tokens", "ner_tags"])

    
    print("Loading model...")
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
        weight_decay=0.01, #changed back from 0.1
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
        compute_metrics=compute_metrics
    )
    # Train
    print("Starting training...")
    trainer.train()

    # Save
    print(f"Saving model to {save_dir}...")
    save_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))

    print("Done!")


if __name__ == "__main__":
    main()
