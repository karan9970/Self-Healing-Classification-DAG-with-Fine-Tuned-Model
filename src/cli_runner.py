import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from inference_node import InferenceNode
from confidence_node import ConfidenceNode
from fallback_node import FallbackNode

class CLIClassifier:
    def __init__(self, model_path="./fine_tuned_model", log_file="./logs/classification_log.json"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.id2label = {0: "negative", 1: "positive"}
        self.log_file = log_file
        self.logs = []

        self.inference = InferenceNode(self.model, self.tokenizer, self.id2label)
        self.confidence = ConfidenceNode(threshold=0.65)
        self.fallback = FallbackNode()

    def classify(self, text: str):
        state = self.inference.run(text)
        state = self.confidence.run(state)

        if state["fallback_triggered"]:
            state = self.fallback.run(state)

        # Logging
        self.logs.append(state)
        with open(self.log_file, "w") as f:
            json.dump(self.logs, f, indent=2)

        print(f"\n✅ Final Label: {state['final_label'].upper()} | Confidence: {state['final_confidence']:.1%}")
        return state

    def run_cli(self):
        print("🚀 Self-Healing Classifier CLI")
        while True:
            text = input("\nEnter text (or 'quit' to exit): ")
            if text.lower() in ["quit", "exit"]:
                break
            self.classify(text)


if __name__ == "__main__":
    classifier = CLIClassifier()
    classifier.run_cli()
