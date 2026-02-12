import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from util.util import resource_path
class Classifier:
    def __init__(self):
        self.saved_dir = resource_path("classifier")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer_loaded = None
        self.model_loaded = None
        self.labled_map = {0:"non bible", 1:"bible"}
        print(f"Classifier initialized on device: {self.device}")
        
    def load_classifier(self):
        self.tokenizer_loaded = AutoTokenizer.from_pretrained(self.saved_dir)
        self.model_loaded = AutoModelForSequenceClassification.from_pretrained(self.saved_dir)
        self.model_loaded.to(self.device)
        print("✅ Classifier loaded.")
        
    def classify(self, text):
        if self.tokenizer_loaded is None or self.model_loaded is None:
            raise Exception("Classifier not loaded. Call load_classifier() first.")
        
        inputs = self.tokenizer_loaded(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            return_token_type_ids=False
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model_loaded(**inputs)
        
        logits = outputs.logits
        
        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        predicted_class_id = torch.argmax(logits, dim=1).item()
        confidence = probabilities[0, predicted_class_id].item()
        
        return self.labled_map[predicted_class_id], confidence
    
    def offload_classifier(self):
        if self.model_loaded is not None:
            self.model_loaded.cpu()
            self.model_loaded = None
        if self.tokenizer_loaded is not None:
            self.tokenizer_loaded = None