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
        
        # Handle both single text and batch (list of texts)
        is_batch = isinstance(text, list)
        texts = text if is_batch else [text]
        
        inputs = self.tokenizer_loaded(
            texts,
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
        
        predicted_class_ids = torch.argmax(logits, dim=1)
        confidences = probabilities[torch.arange(len(texts)), predicted_class_ids]
        
        # Return in same format as input
        if is_batch:
            return [(self.labled_map[cid.item()], conf.item()) 
                    for cid, conf in zip(predicted_class_ids, confidences)]
        else:
            return self.labled_map[predicted_class_ids[0].item()], confidences[0].item()
    
    def offload_classifier(self):
        if self.model_loaded is not None:
            self.model_loaded.cpu()
            self.model_loaded = None
        if self.tokenizer_loaded is not None:
            self.tokenizer_loaded = None