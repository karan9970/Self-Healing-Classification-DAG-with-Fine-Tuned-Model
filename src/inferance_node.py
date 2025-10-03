import torch
from datetime import datetime

class InferenceNode:
    def __init__(self, model, tokenizer, id2label):
        self.model = model
        self.tokenizer = tokenizer
        self.id2label = id2label
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def run(self, user_input: str):
        """Run model inference on user input"""
        inputs = self.tokenizer(
            user_input,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)[0]

        predicted_idx = torch.argmax(probabilities).item()
        predicted_label = self.id2label[predicted_idx]
        confidence = probabilities[predicted_idx].item()

        return {
            "user_input": user_input,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "all_probabilities": {
                self.id2label[i]: probabilities[i].item() for i in range(len(probabilities))
            },
            "timestamp": datetime.now().isoformat()
        }
