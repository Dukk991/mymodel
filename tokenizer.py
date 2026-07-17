class Tokenizer:
    def __init__(self, text):
        self.words = sorted(set(text.split()))
    
        self.words_to_ids = { w: i for i, w in enumerate(self.words) }

        self.ids_to_words = { i: w for w, i in self.words_to_ids.items() }
    
    def encode(self, text):
        return [self.words_to_ids[w] for w in text.split()]

    def decode(self, ids):
        return " ".join(self.ids_to_words[i] for i in ids)
