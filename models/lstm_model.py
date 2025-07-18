import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class TextDataset(Dataset):
    def __init__(self, sequences, sequence_length):
        self.sequences = sequences
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        input_seq = torch.tensor(sequence[:-1], dtype=torch.long)
        target_seq = torch.tensor(sequence[1:], dtype=torch.long)
        return input_seq, target_seq

class LSTMTextGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
        super(LSTMTextGenerator, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden=None):
        # Embedding
        embedded = self.embedding(x)
        
        # LSTM
        lstm_out, hidden = self.lstm(embedded, hidden)
        
        # Dropout and fully connected
        output = self.dropout(lstm_out)
        output = self.fc(output)
        
        return output, hidden
    
    def generate_text(self, preprocessor, start_text="", max_length=100, temperature=1.0):
        """Generate text using the trained model"""
        self.eval()
        
        if start_text:
            sequence = preprocessor.text_to_sequence(start_text)
        else:
            sequence = [preprocessor.vocab['<START>']]
        
        hidden = None
        generated = sequence.copy()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Prepare input
                input_seq = torch.tensor([sequence], dtype=torch.long)
                
                # Forward pass
                output, hidden = self.forward(input_seq, hidden)
                
                # Get last output and apply temperature
                logits = output[0, -1, :] / temperature
                probabilities = torch.softmax(logits, dim=0)
                
                # Sample next word
                next_word_idx = torch.multinomial(probabilities, 1).item()
                
                # Stop if end token
                if next_word_idx == preprocessor.vocab.get('<END>', -1):
                    break
                
                generated.append(next_word_idx)
                sequence = [next_word_idx]  # Use only last word for next prediction
        
        return preprocessor.sequence_to_text(generated)

def train_lstm_model(model, dataloader, num_epochs=10, learning_rate=0.001):
    """Train the LSTM model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    losses = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (input_seq, target_seq) in enumerate(dataloader):
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output, _ = model(input_seq)
            
            # Calculate loss
            loss = criterion(output.reshape(-1, model.vocab_size), target_seq.reshape(-1))
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    return losses