import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from util.util import resource_path
def mask_entities(text):
    import re
    PERSON_PATTERN = re.compile(
        r'\b(Jesus|Christ|Moses|Abraham|Isaac|Jacob|David|Solomon|'
        r'Elijah|Elisha|Paul|Peter|John|Mary|Joseph|Noah|Daniel|'
        r'Isaiah|Jeremiah|Ezekiel|Joshua|Samuel|Saul|Aaron|Ruth|'
        r'Esther|Ezra|Nehemiah|Job|Jonah|Micah|Amos|Hosea|Joel|'
        r'Obadiah|Nahum|Habakkuk|Zephaniah|Haggai|Zechariah|Malachi|'
        r'Barnabas|Timothy|Titus|Silas|Stephen|Philip|Andrew|James|'
        r'Thomas|Matthew|Luke|Mark|Lazarus|Martha|Nicodemus|Pilate|'
        r'Herod|Adam|Eve|Cain|Abel|Enoch|Lot|Rebekah|Leah|Rachel|'
        r'Miriam|Deborah|Gideon|Samson|Delilah|Boaz|Hannah|Eli|'
        r'Jonathan|Goliath|Bathsheba|Absalom|Rehoboam|Jeroboam|'
        r'Ahab|Jezebel|Gehazi|Joash|Hezekiah|Josiah|'
        r'Zerubbabel|Mordecai|Haman|Cyrus|Darius|Nebuchadnezzar)\b',
        re.IGNORECASE
    )
    GOD_TITLE_PATTERN = re.compile(
        r'\b(Lord|God|Father|Holy Spirit|Holy Ghost|Almighty|Most High|'
        r'Jehovah|Yahweh|Elohim|Adonai|El Shaddai|Emmanuel|Immanuel|'
        r'Messiah|Saviour|Savior|Redeemer|Creator|Sovereign)\b',
        re.IGNORECASE
    )
    PLACE_PATTERN = re.compile(
        r'\b(Jerusalem|Israel|Judah|Babylon|Egypt|Canaan|Galilee|'
        r'Nazareth|Bethlehem|Zion|Jordan|Sinai|Samaria|Jericho|'
        r'Capernaum|Gethsemane|Calvary|Golgotha|Bethany|Emmaus|'
        r'Damascus|Antioch|Corinth|Ephesus|Philippi|Thessalonica|'
        r'Galatia|Rome|Athens|Judea|Hebron|Bethel|'
        r'Gilead|Moab|Edom|Assyria|Persia|Macedonia|Arabia)\b',
        re.IGNORECASE
    )
    text = PERSON_PATTERN.sub('[person]', text)
    text = GOD_TITLE_PATTERN.sub('[god_title]', text)
    text = PLACE_PATTERN.sub('[place]', text)
    return text
class Classifier:
    def __init__(self):
        self.saved_dir = resource_path("classifier")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer_loaded = None
        self.model_loaded = None
        self.label_map = {0: "non bible", 1: "bible"}
        print(f"Classifier initialized on device: {self.device}")

    def load_classifier(self):
        self.tokenizer_loaded = AutoTokenizer.from_pretrained(self.saved_dir)
        self.model_loaded = AutoModelForSequenceClassification.from_pretrained(self.saved_dir)
        self.model_loaded = self.model_loaded.float().to(self.device)
        self.model_loaded.eval()
        print("✅ Classifier loaded.")

    def classify(self, text):
        if self.tokenizer_loaded is None or self.model_loaded is None:
            raise Exception("Classifier not loaded. Call load_classifier() first.")

        is_batch = isinstance(text, list)
        texts = text if is_batch else [text]

        texts = [mask_entities(t) for t in texts]

        inputs = self.tokenizer_loaded(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model_loaded(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = outputs.logits.argmax(-1)

        if is_batch:
            return [
                (self.label_map[p.item()], probs[i][p.item()].item())
                for i, p in enumerate(preds)
            ]
        else:
            pred = preds[0].item()
            return [(self.label_map[pred], probs[0][pred].item())]

    def offload_classifier(self):
        if self.model_loaded is not None:
            self.model_loaded.cpu()
            self.model_loaded = None
        if self.tokenizer_loaded is not None:
            self.tokenizer_loaded = None