# text-generation-project

This project is designed for text generation using machine learning models. It includes implementations of LSTM and GPT-based models, along with utilities for data preprocessing and a demonstration notebook.

## Project Structure

- **data/**: Contains sample texts used for training or testing the text generation models.
  - `sample_texts.txt`: Sample texts for model training and testing.

- **models/**: Contains the model implementations.
  - `lstm_model.py`: Implements an LSTM-based text generation model.
  - `gpt_model.py`: Implements a GPT-based text generation model.

- **notebooks/**: Contains Jupyter notebooks for demonstration purposes.
  - `text_generation_demo.ipynb`: Demonstrates how to use the text generation models.

- **utils/**: Contains utility functions for text preprocessing.
  - `text_preprocessing.py`: Functions for tokenization, normalization, and dataset creation.

- **requirements.txt**: Lists the dependencies required for the project.

- **main.py**: The entry point for the application, initializing models and running the text generation process.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd text-generation-project
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the text generation process, execute the following command:
```
python main.py
```

## Models

- **LSTM Model**: A recurrent neural network model that is capable of learning long-term dependencies in sequential data.
- **GPT Model**: A transformer-based model that generates text based on the input context.

## License

This project is licensed under the MIT License.
