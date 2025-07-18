import re
import nltk
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class TextPreprocessor:
    def __init__(self):
        self.vocab = {}
        self.reverse_vocab = {}
        self.vocab_size = 0
    
    def clean_text(self, text):
        """Clean and normalize text"""
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Keep only letters, numbers, and basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.,!?;:]', '', text)
        return text.strip()
    
    def tokenize(self, text):
        """Tokenize text into words"""
        return nltk.word_tokenize(text)
    
    def build_vocabulary(self, texts, min_freq=2):
        """Build vocabulary from texts"""
        all_words = []
        for text in texts:
            cleaned_text = self.clean_text(text)
            tokens = self.tokenize(cleaned_text)
            all_words.extend(tokens)
        
        # Count word frequencies
        word_counts = Counter(all_words)
        
        # Build vocabulary with minimum frequency threshold
        self.vocab = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
        idx = 4
        
        for word, count in word_counts.items():
            if count >= min_freq:
                self.vocab[word] = idx
                idx += 1
        
        self.reverse_vocab = {idx: word for word, idx in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        
        print(f"Vocabulary size: {self.vocab_size}")
        return self.vocab
    
    def text_to_sequence(self, text):
        """Convert text to sequence of indices"""
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize(cleaned_text)
        sequence = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        return sequence
    
    def sequence_to_text(self, sequence):
        """Convert sequence of indices back to text"""
        words = [self.reverse_vocab.get(idx, '<UNK>') for idx in sequence]
        return ' '.join(words)
    
    def create_sequences(self, texts, sequence_length=50):
        """Create input-output sequence pairs for training"""
        sequences = []
        for text in texts:
            seq = self.text_to_sequence(text)
            for i in range(len(seq) - sequence_length):
                sequences.append(seq[i:i + sequence_length + 1])
        return sequences