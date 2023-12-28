# Neural Machine Translation (NMT) System

## Overview
This repository contains the implementation of a Neural Machine Translation (NMT) system, specifically designed for translating between English and Vietnamese. It uses a deep learning approach to understand and convert text from one language to another, harnessing the power of Python and PyTorch, along with the sequence-to-sequence (seq2seq) model and attention mechanisms.

## Features
- **Encoder-Decoder Architecture:** Utilizes a seq2seq model where the encoder processes the input language and the decoder generates the translated output.
- **Attention Mechanism:** Implements Bahdanau's attention mechanism for improved translation quality, allowing the model to focus on different parts of the input sequence dynamically.
- **Data Preprocessing:** Comprehensive preprocessing steps including normalization, contraction expansion, and filtering to prepare data for optimal training and translation outcomes.
- **Bucketing Strategy:** Implements a bucketing strategy to group sentences of similar lengths together, improving training efficiency and performance.
- **Training and Evaluation Utilities:** Includes scripts for training the model with custom data, evaluating translations, and calculating BLEU scores for performance measurement.

## Installation

Before running the project, ensure that you have the following prerequisites installed:
- Python 3.x
- PyTorch
- NLTK

To set up the project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/YourRepository/NeuralMachineTranslation.git
    cd NeuralMachineTranslation
    ```

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The main functionalities include training the model, testing its performance on a dataset, and translating individual sentences. Here's how to use each:

### Training the Model
To train the model with your dataset, ensure your data is formatted correctly and run:
```bash
python main.py train
```

### Testing the Model
To test the model on a test dataset and get a BLEU score evaluation, run:
```bash
python main.py test
```

### Translating Text
To translate a specific sentence, run:
```bash
python main.py translate --sentence "Your sentence here"
```

### Development Mode
To randomly evaluate sentences from your dataset, you can use the development mode:
```bash
python main.py dev
```

### Customization
To adapt this project to different language pairs or datasets:

1. Language Pairs: Modify the Lang class instances and preprocessing steps to       fit the specific characteristics of your languages.
2. Model Parameters: Adjust the hidden_size, encoder, and decoder definitions in    the main script to experiment with different model capacities and structures.
3. Dataset: Replace the training and testing datasets with your desired language    pair data, ensuring it's formatted as expected by the prepareData function.

Happy Translating! 🌐


