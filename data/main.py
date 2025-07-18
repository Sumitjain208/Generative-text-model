import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.text_preprocessing import TextPreprocessor
from models.lstm_model import LSTMTextGenerator, TextDataset, train_lstm_model
from models.gpt_model import GPTTextGenerator, fine_tune_gpt_model
from torch.utils.data import DataLoader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def load_sample_data():
    """Load sample text data"""
    data_path = 'data/sample_texts.txt'
    
    if not os.path.exists(data_path):
        print("Sample data not found. Creating sample data...")
        os.makedirs('data', exist_ok=True)
        
        sample_texts = [
            "Technology is rapidly evolving and changing our daily lives. Artificial intelligence and machine learning are becoming integral parts of modern society.",
            "Climate change represents one of the most pressing challenges of our time. Rising temperatures affect weather patterns worldwide.",
            "Space exploration continues to fascinate humanity and drive scientific advancement. Recent missions to Mars have provided valuable insights.",
            "Education systems worldwide are adapting to digital transformation. Online learning platforms provide accessible education to students globally.",
            "Healthcare innovation saves lives and improves quality of life for millions. Medical research leads to breakthrough treatments."
        ]
        
        with open(data_path, 'w') as f:
            f.write('\n\n'.join(sample_texts))
    
    with open(data_path, 'r') as f:
        text = f.read()
    
    # Split into paragraphs
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    return paragraphs

def demonstrate_lstm_generation():
    """Demonstrate LSTM text generation"""
    print("=== LSTM Text Generation Demo ===")
    
    # Load data
    texts = load_sample_data()
    
    # Preprocess
    preprocessor = TextPreprocessor()
    preprocessor.build_vocabulary(texts)
    
    # Create sequences
    sequences = preprocessor.create_sequences(texts, sequence_length=20)
    
    # Create dataset and dataloader
    dataset = TextDataset(sequences, sequence_length=20)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    model = LSTMTextGenerator(
        vocab_size=preprocessor.vocab_size,
        embedding_dim=64,
        hidden_dim=128,
        num_layers=2
    )
    
    # Train model
    print("Training LSTM model...")
    losses = train_lstm_model(model, dataloader, num_epochs=5)
    
    # Generate text
    print("\nGenerating text with LSTM...")
    
    prompts = [
        "Technology",
        "Climate change",
        "Space exploration",
        "Education",
        "Healthcare"
    ]
    
    for prompt in prompts:
        generated = model.generate_text(preprocessor, start_text=prompt, max_length=50)
        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: {generated}")
    
    return model, preprocessor, losses

def demonstrate_gpt_generation():
    """Demonstrate GPT text generation"""
    print("\n=== GPT Text Generation Demo ===")
    
    # Initialize GPT model
    gpt_generator = GPTTextGenerator('gpt2')
    
    # Generate text
    prompts = [
        "Technology is revolutionizing",
        "Climate change impacts",
        "Space exploration reveals",
        "Modern education systems",
        "Healthcare innovations"
    ]
    
    for prompt in prompts:
        generated_texts = gpt_generator.generate_text(
            prompt=prompt,
            max_length=80,
            temperature=0.8,
            num_return_sequences=1
        )
        
        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: {generated_texts[0]}")
    
    return gpt_generator

def interactive_generation(lstm_model=None, preprocessor=None, gpt_generator=None):
    """Interactive text generation"""
    print("\n=== Interactive Text Generation ===")
    print("Enter prompts to generate text. Type 'quit' to exit.")
    
    while True:
        prompt = input("\nEnter your prompt: ").strip()
        
        if prompt.lower() == 'quit':
            break
        
        if not prompt:
            continue
        
        print("\nChoose model:")
        print("1. LSTM")
        print("2. GPT")
        
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == '1' and lstm_model and preprocessor:
            generated = lstm_model.generate_text(
                preprocessor, 
                start_text=prompt, 
                max_length=60,
                temperature=0.8
            )
            print(f"\nLSTM Generated: {generated}")
        
        elif choice == '2' and gpt_generator:
            generated_texts = gpt_generator.generate_text(
                prompt=prompt,
                max_length=80,
                temperature=0.8
            )
            print(f"\nGPT Generated: {generated_texts[0]}")
        
        else:
            print("Invalid choice or model not available.")

def main():
    """Main function"""
    print("Text Generation Model Demo")
    print("=" * 40)
    
    # Demonstrate LSTM
    lstm_model, preprocessor, losses = demonstrate_lstm_generation()
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('LSTM Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
    
    # Demonstrate GPT
    gpt_generator = demonstrate_gpt_generation()
    
    # Interactive generation
    interactive_generation(lstm_model, preprocessor, gpt_generator)

if __name__ == "__main__":
    main()