"""Fine-tune GLiNER large v2.1 for NER on LLM generated conll data -- using Urchade's hyperparameters."""
## disclaimer: these files were created by the data scientists by asking claude to convert my fine-tuning notebook into a script that can run on their virtual machine

import argparse
import os
from pathlib import Path

from gliner import GLiNER
from gliner.training import Trainer, TrainingArguments


MODEL_NAME = "models/gliner_large-v2.1"
OUTPUT_DIR = Path("gliner1_finetuned_on_llm")
SAVE_DIR = Path("finetuned_gliner1_llm")
TRAIN_FILE = Path("data/LLM_TRAINING.conll")
VAL_FILE = Path("data/LLM_VALIDATION.conll")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output/save directory (e.g. Azure ML output mount)")
    parser.add_argument("--fp16", action="store_true",
                        help="Enable mixed-precision training")
    return parser.parse_args()


def load_conll(file):
    sentences = []
    current_sentence = []

    with open(file, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()

            if line == "":
                if len(current_sentence) > 0:
                    sentences.append(current_sentence)
                    current_sentence = []
            elif line.startswith("-DOCSTART-"):
                continue
            else:
                parts = line.split("\t")

                if len(parts) == 2:
                    token = parts[0]
                    label = parts[1]
                    current_sentence.append((token, label))
                else:
                    print(f"missed line!{repr(line)} ")

        if len(current_sentence) > 0:
            sentences.append(current_sentence)
    return sentences


def bio_to_spans(sentence):
    """this converst BIO labels into GLiNER entity spans."""
    tokens = []
    spans = []
    for token, label in sentence: # we get all the tokens from the sent
        tokens.append(token)
    current_start = None #start index of the current entity tracked
    current_label = None #entity type of the entity tracked
    for token_index, (token, label) in enumerate(sentence): #get each token's bio label
        if label.startswith("B-"): #if a new ent starts,
            if current_label is not None: #if another ent is being tracked, close it first
                spans.append([current_start, token_index - 1, current_label]) #save the start and end index and label!
            current_start = token_index #setting the start for the new entity
            current_label = label[2:] #remove bio prefix for ent type
        elif label.startswith("I-"): 
            entity_type = label[2:] #same here
            if current_label == entity_type: #if same ent type, continue ent
                continue
            if current_label is not None: #if another ent is tracked, close it first
                spans.append([current_start, token_index - 1, current_label]) #saving the previous ent
            current_start = token_index #start the new ent
            current_label = entity_type #and set the new ent type
        else:
            if current_label is not None: #if ent tracked, close it
                spans.append([current_start, token_index - 1, current_label]) #save the ent
                current_start = None #reset the ent start
                current_label = None #reset the ent type
    if current_label is not None: #if an ent continues until the end, we close it
        spans.append([current_start, len(tokens) - 1, current_label]) #using the final token as end index
    return tokens, spans

def sentences2glinerdict(sentences):
    """this converts a list of CoNLL sentences to a list of GLiNER-format dicts."""
    gliner_data = []

    for sentence in sentences:
        tokens, spans = bio_to_spans(sentence) #convert the bio labels into spans
        gliner_data.append({"tokenized_text": tokens, "ner": spans}) #save tokens and spans for special gliner formtting
    return gliner_data


def filter_empty_spans(gliner_data):
    """this removes sentences with no entity spans since gliner's spandatacollator cannot process zero-span examples"""
    filtered_data = []
    for example in gliner_data:
        if len(example["ner"]) > 0: #keep the example if the labels contain at least one entity span
            filtered_data.append(example) #add to filtered data
    return filtered_data



# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def finetune_gliner(train_data, val_data, model_name, output_dir, fp16=False):
    """Finetunes gliner and returns the best checkpoint."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading base model: {model_name}")
    model = GLiNER.from_pretrained(model_name)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=5e-6,
        weight_decay=0.01,
        others_lr=1e-5,
        others_weight_decay=0.01,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=6,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=67,
        fp16=fp16,
    )

    data_collator = SpanDataCollator(
        config=model.config,
        data_processor=model.data_processor,
        prepare_labels=True,
        prepare_entities=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        processing_class=model.data_processor.transformer_tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    return model


def main():
    args = parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    save_dir = output_dir / "finetuned_gliner1_llm" if args.output_dir else SAVE_DIR

    print(f"Train file : {TRAIN_FILE}")
    print(f"Val file   : {VAL_FILE}")
    print(f"Output dir : {output_dir}")
    print(f"Save dir   : {save_dir}")
    print()

    print("Loading data...")
    train_sentences = load_conll(TRAIN_FILE)
    val_sentences = load_conll(VAL_FILE)

    train_gliner = sentences2glinerdict(train_sentences)
    val_gliner = sentences2glinerdict(val_sentences)

    print(f"Train examples: {len(train_gliner)}")
    print(f"Val   examples: {len(val_gliner)}")
    print()

    finetuned_model = finetune_gliner(
        train_gliner, val_gliner, MODEL_NAME, str(output_dir), fp16=args.fp16,
    )

    finetuned_model.eval()

    save_dir.mkdir(parents=True, exist_ok=True)
    finetuned_model.save_pretrained(str(save_dir))
    print(f"Done. Best model saved to: {save_dir}")


if __name__ == "__main__":
    main()
