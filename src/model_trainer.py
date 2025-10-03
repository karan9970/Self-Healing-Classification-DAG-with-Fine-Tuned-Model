# src/model_trainer.py
import torch
import numpy as np
import json
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class ModelTrainer:
    def __init__(self, model_name="distilbert-base-uncased", output_dir="./fine_tuned_model"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Using device: {self.device}")

    def load_and_prepare_data(self, max_samples=3000):
        print("\n📊 Loading IMDB dataset...")
        dataset = load_dataset("imdb")

        train_dataset = dataset["train"].shuffle(seed=42).select(range(max_samples))
        test_dataset = dataset["test"].shuffle(seed=42).select(range(max_samples // 5))

        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=256
            )

        train_dataset = train_dataset.map(tokenize_function, batched=True)
        test_dataset = test_dataset.map(tokenize_function, batched=True)

        return train_dataset, test_dataset

    def create_lora_model(self):
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
            id2label={0: "negative", 1: "positive"},
            label2id={"negative": 0, "positive": 1}
        )

        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_lin", "v_lin"],
            bias="none"
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        return model

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary'
        )
        accuracy = accuracy_score(labels, predictions)

        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    def train(self):
        train_dataset, test_dataset = self.load_and_prepare_data()
        model = self.create_lora_model()

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=2,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            push_to_hub=False,
            report_to="none"
        )

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )

        train_result = trainer.train()
        eval_results = trainer.evaluate()

        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

        training_info = {
            "model_name": self.model_name,
            "training_date": datetime.now().isoformat(),
            "train_samples": len(train_dataset),
            "test_samples": len(test_dataset),
            "final_metrics": eval_results,
            "training_time": train_result.metrics['train_runtime']
        }

        with open(f"{self.output_dir}/training_info.json", "w") as f:
            json.dump(training_info, f, indent=2)

        return model, eval_results

