from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import time
import math
from torch.utils.data import BatchSampler, SequentialSampler
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import argparse
import sys

# Constants for special tokens and model parameters
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 50
hidden_size = 128
batch_size = 32

# Patterns for expanding contractions in English text (e.g., "can't" to "cannot")
contraction_patterns = [
    (r"can't", "cannot"),
    (r"can 't", "cannot"),
    (r"won't", "will not"),
    (r"won 't", "will not"),
    (r"n't", " not"),
    (r"'re", " are"),
    (r"'s", " is"),
    (r"'d", " would"),
    (r"'ll", " will"),
    (r"'t", " not"),
    (r"'ve", " have"),
    (r"'m", " am"),
    (r"'d", " had"),
]

# Language class to manage vocabulary and encoding/decoding words
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"<UNK>": 0, "SOS": 1, "EOS": 2}
        self.word2count = {"<UNK>": 0}
        self.index2word = {0: "<UNK>", 1: "SOS", 2: "EOS"}
        self.n_words = 3  # Count UNK, SOS, and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Function to expand contractions in text
def expand_contractions(s, contraction_patterns=contraction_patterns):
    for pattern, replacement in contraction_patterns:
        s = re.sub(pattern, replacement, s)
    return s

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"&apos;", "'", s)
    s = expand_contractions(s)
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?']+ ", r" ", s)
    return s.strip()

# Normalize and filter out unknown words
def normalizeAndFilterUnknowns(s, lang):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"&apos;", "'", s)
    s = expand_contractions(s, contraction_patterns)
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?']+ ", r" ", s)
    return ' '.join(word if word in lang.word2index else '<UNK>' for word in s.strip().split())


#Ensure pair is shorter than max length
def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2, filename1, filename2, reverse=False,verbose=False):
    # Create Lang instances for both languages
    input_lang = Lang(lang1)
    output_lang = Lang(lang2)

    # Read the files associated with each language and split the text into lines
    lines_lang1 = open(filename1, encoding='utf-8').read().strip().split('\n')
    lines_lang2 = open(filename2, encoding='utf-8').read().strip().split('\n')

    # Check if the files for both languages have the same number of lines
    if len(lines_lang1) != len(lines_lang2):
        raise ValueError("The number of lines in both files must be the same")

    # Pair up corresponding sentences from the two language files
    pairs = [[normalizeString(s1), normalizeString(s2)] for s1, s2 in zip(lines_lang1, lines_lang2)]

    # If reverse is true, reverse the sentence pairs and swap the roles of input and output languages
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang, output_lang = output_lang, input_lang

    # If verbose is true, print the number of sentence pairs read
    if verbose:
        print("Read %s sentence pairs" % len(pairs))

    # Filter the pairs (based on criteria defined in filterPairs function)
    pairs = filterPairs(pairs)        

    # Add each sentence in the pairs to the respective language instances
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])

    # If verbose is true, print the number of sentence pairs after filtering and the word count in each language
    if verbose:
        print("Trimmed to %s sentence pairs" % len(pairs))
        print("Counted words:")
        print(input_lang.name, input_lang.n_words)
        print(output_lang.name, output_lang.n_words)
    
    # Return the language instances and the final list of sentence pairs
    return input_lang, output_lang, pairs



class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size  # Store the size of the hidden layer

        self.embedding = nn.Embedding(input_size, hidden_size)  # Embedding layer to transform input indices into dense vectors of a fixed size
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)  # GRU layer for processing sequences
        self.dropout = nn.Dropout(dropout_p)  # Dropout layer for regularization

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))  # Apply dropout to the embedded input
        output, hidden = self.gru(embedded)  # Process the input through the GRU layer
        return output, hidden  # Return the output and the hidden state


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)  # Embedding layer for the output indices
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)  # GRU layer for processing sequences
        self.out = nn.Linear(hidden_size, output_size)  # Linear layer to transform the GRU output to the desired output size

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)  # Determine the batch size from encoder outputs
        # Initialize the decoder input with the start-of-sequence token
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden  # Use the encoder's hidden state to start the decoder
        decoder_outputs = []  # Store the decoder's outputs

        for i in range(MAX_LENGTH):
            # Process each step through the decoder
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: use the target tensor for the next input
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                # No teacher forcing: use the decoder's own output as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # Detach from history as input

        # Combine the outputs and apply log softmax for the final output
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None  # Return outputs, hidden state, and None for consistency

    def forward_step(self, input, hidden):
        # A single forward step of the decoder
        output = self.embedding(input)  # Embed the input
        output = F.relu(output)  # Apply ReLU activation
        output, hidden = self.gru(output, hidden)  # Process through the GRU
        output = self.out(output)  # Transform to the output size
        return output, hidden  # Return the output and updated hidden state

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)  # Linear layer for the query (decoder hidden state)
        self.Ua = nn.Linear(hidden_size, hidden_size)  # Linear layer for the keys (encoder outputs)
        self.Va = nn.Linear(hidden_size, 1)  # Linear layer to compute the attention scores

    def forward(self, query, keys):
        # Compute attention scores
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))  # Apply the attention mechanism
        scores = scores.squeeze(2).unsqueeze(1)  # Reshape scores to have the proper dimensions

        weights = F.softmax(scores, dim=-1)  # Apply softmax to get attention weights
        context = torch.bmm(weights, keys)  # Compute the weighted sum of encoder outputs

        return context, weights  # Return the context vector and attention weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)  # Embedding layer for output indices
        self.attention = BahdanauAttention(hidden_size)  # Bahdanau attention mechanism
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)  # GRU layer that considers both context and embedded input
        self.out = nn.Linear(hidden_size, output_size)  # Linear layer for final output size
        self.dropout = nn.Dropout(dropout_p)  # Dropout for regularization

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)  # Determine the batch size
        # Initialize decoder input with the start-of-sequence token
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden  # Initialize decoder hidden state with encoder's hidden state
        decoder_outputs = []  # Store decoder outputs
        attentions = []  # Store attention weights

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: use the target tensor for the next input
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                # No teacher forcing: use the decoder's own output as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # Detach from history as input

        # Combine decoder outputs and apply log softmax for the final output
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)  # Combine attention weights

        return decoder_outputs, decoder_hidden, attentions

    def forward_step(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input))  # Embed the input and apply dropout

        query = hidden.permute(1, 0, 2)  # Rearrange hidden state to match attention layer's expectation
        context, attn_weights = self.attention(query, encoder_outputs)  # Get context and attention weights
        input_gru = torch.cat((embedded, context), dim=2)  # Concatenate embedded input and context

        output, hidden = self.gru(input_gru, hidden)  # Process through GRU
        output = self.out(output)  # Transform to the output size

        return output, hidden, attn_weights  # Return output, updated hidden state, and attention weights


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] if word in lang.word2index else lang.word2index["<UNK>"] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

# Create buckets of pairs based on sentence lengths
def create_buckets(pairs, bucket_size):
    # Sort pairs by the length of the input sentence
    pairs.sort(key=lambda p: len(p[0].split(' ')))

    buckets = []
    current_bucket = []
    current_length = len(pairs[0][0].split(' '))

    for pair in pairs:
        if len(pair[0].split(' ')) > current_length + bucket_size:
            buckets.append(current_bucket)
            current_bucket = []
            current_length = len(pair[0].split(' '))

        current_bucket.append(pair)

    # Add the last bucket
    if current_bucket:
        buckets.append(current_bucket)

    return buckets

#Show how sentences have been divided to speed up training
def display_buckets(buckets):
    for i, bucket in enumerate(buckets):
        if bucket:
            min_length = min(len(pair[0].split(' ')) for pair in bucket)
            max_length = max(len(pair[0].split(' ')) for pair in bucket)
            print(f"Bucket {i+1}: {len(bucket)} pairs, Length range: {min_length}-{max_length}")

def get_dataloader(batch_size, buckets, input_lang, output_lang):
    dataloaders = []  # List to store the data loaders

    for bucket in buckets:
        # Initialize arrays for input and target ids with maximum length and filled with zeros
        input_ids = np.zeros((len(bucket), MAX_LENGTH), dtype=np.int32)
        target_ids = np.zeros((len(bucket), MAX_LENGTH), dtype=np.int32)

        for idx, (inp, tgt) in enumerate(bucket):
            # Convert sentences to index arrays
            inp_ids = indexesFromSentence(input_lang, inp)
            tgt_ids = indexesFromSentence(output_lang, tgt)
            inp_ids.append(EOS_token)  # Append End-of-Sentence token
            tgt_ids.append(EOS_token)  # Append End-of-Sentence token
            # Fill the arrays with sentence indices
            input_ids[idx, :len(inp_ids)] = inp_ids
            target_ids[idx, :len(tgt_ids)] = tgt_ids

        # Create a tensor dataset and dataloader for the bucket
        train_data = TensorDataset(torch.LongTensor(input_ids).to(device),
                                   torch.LongTensor(target_ids).to(device))
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        dataloaders.append(train_dataloader)

    return input_lang, output_lang, dataloaders  # Return the data loaders

def train_epoch(dataloaders, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    total_loss = 0  # Initialize total loss for the epoch

    for dataloader in dataloaders:
        for input_tensor, target_tensor in dataloader:
            # Zero the gradients of both optimizers
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            # Forward pass through the encoder and decoder
            encoder_outputs, encoder_hidden = encoder(input_tensor)
            decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

            # Compute loss
            loss = criterion(
                decoder_outputs.view(-1, decoder_outputs.size(-1)),
                target_tensor.view(-1)
            )
            loss.backward()  # Backpropagate error

            # Update the weights
            encoder_optimizer.step()
            decoder_optimizer.step()

            total_loss += loss.item()  # Accumulate the loss

    return total_loss  # Return the total loss for the epoch


#For timing training
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

#For timing training
def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def train(dataloaders, encoder, decoder, n_epochs, learning_rate=0.001, print_every=100):
    start = time.time()  # Record the start time for calculating elapsed time
    print_loss_total = 0  # Initialize total loss for printing

    # Initialize optimizers for encoder and decoder
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()  # Negative log likelihood loss

    for epoch in range(1, n_epochs + 1):
        # Train for one epoch and get the loss
        loss = train_epoch(dataloaders, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss  # Accumulate loss

        if epoch % print_every == 0:
            # Print average loss every 'print_every' epochs
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0  # Reset total loss
            # Print time elapsed and loss
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                        epoch, epoch / n_epochs * 100, print_loss_avg))

def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():  # Disable gradient calculations for evaluation
        # Convert the input sentence to tensor
        input_tensor = tensorFromSentence(input_lang, normalizeAndFilterUnknowns(sentence, input_lang))

        # Forward pass through the encoder
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        # Forward pass through the decoder
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)  # Get the most probable next words
        decoded_ids = topi.squeeze()  # Remove unnecessary dimensions

        decoded_words = []  # List to store the decoded words
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_words.append('<EOS>')  # End of sentence token
                break
            decoded_words.append(output_lang.index2word[idx.item()])  # Append the word to the list
    return decoded_words, decoder_attn  # Return the decoded sentence and attention weights


#Pick n sentences randomly and show their translations
def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, _ = evaluate(encoder, decoder, pair[0], input_lang, output_lang)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

def translateSentence(encoder, decoder, sentence):
    print("Input: " + sentence)
    output_words, _ = evaluate(encoder, decoder, sentence, input_lang, output_lang)
    output_sentence = ' '.join(output_words)
    print('Output: ', output_sentence)

def test_model(input_lang, output_lang, test_pairs, encoder, decoder, device):
    encoder.eval()  # Set the encoder to evaluation mode
    decoder.eval()  # Set the decoder to evaluation mode
    references = []  # List to store the actual target sentences
    hypotheses = []  # List to store the model's predictions

    for sent in test_pairs:
        references.append([sent[1]])  # Add the actual translation to references
        input_tensor = tensorFromSentence(input_lang, sent[0])  # Convert source sentence to tensor
        
        # Generate translation using the model without calculating gradients
        with torch.no_grad():
            encoder_outputs, encoder_hidden = encoder(input_tensor.to(device))
            decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

            _, topi = decoder_outputs.topk(1)  # Get the highest probability output
            decoded_ids = topi.squeeze()  # Remove extra dimensions

            decoded_words = []  # Store the words in the prediction
            for idx in decoded_ids:
                if idx.item() == EOS_token:  # Break if EOS token is reached
                    break
                decoded_words.append(output_lang.index2word[idx.item()])  # Add the word to the prediction
        
        hypotheses.append(' '.join(decoded_words).replace("<EOS>", "").strip())  # Clean and add the prediction to hypotheses

    # Calculate BLEU score for the predictions
    chencherry = SmoothingFunction()
    bleu_score = corpus_bleu(references, hypotheses, smoothing_function=chencherry.method1)
    print(f'BLEU score: {bleu_score}')

    return bleu_score  # Return the BLEU score


if __name__ == "__main__":
    # Argument parsing for different modes
    parser = argparse.ArgumentParser(description='Train, Test, or Translate english to vietnamese')
    parser.add_argument('mode', choices=['train', 'test', 'translate', 'dev'], help='Train, Test, or Translate mode')
    parser.add_argument('--sentence', '-s', type=str, required=False, help='Sentence to translate')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Prepare data for training or testing
    input_lang, output_lang, pairs = prepareData('viet', 'eng', 'data/train.vi.txt.', 'data/train.en.txt', True, True)

    # Create dataloaders for training
    buckets = create_buckets(pairs, bucket_size=5)
    input_lang, output_lang, train_dataloader = get_dataloader(batch_size, buckets, input_lang, output_lang)

    if args.mode == "train":
        # Training mode
        display_buckets(buckets)
        encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
        decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)

        print("Beginning Training")
        train(train_dataloader, encoder, decoder, 100, print_every=5)
        torch.save(encoder.state_dict(), 'model/encoder.pth')
        torch.save(decoder.state_dict(), 'model/decoder.pth')
        print("Model saved!")

    elif args.mode == "test":
        # Testing mode
        _, _, pairs = prepareData('viet', 'eng', 'data/tst2012.vi.txt.', 'data/tst2012.en.txt', True)
        encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
        decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)
        encoder.load_state_dict(torch.load('model/encoder.pth'))
        decoder.load_state_dict(torch.load('model/decoder.pth'))
        test_model(input_lang, output_lang, pairs, encoder, decoder, device)

    elif args.mode == "translate":
        # Translation mode
        if args.sentence is None:
            print("Please provide a sentence to translate using --sentence")
            sys.exit()
        
        encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
        decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)
        encoder.load_state_dict(torch.load('model/encoder.pth'))
        decoder.load_state_dict(torch.load('model/decoder.pth'))
        translateSentence(encoder, decoder, args.sentence)

    elif args.mode == "dev":
        # Development mode for random evaluation
        encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
        decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)
        encoder.load_state_dict(torch.load('model/encoder.pth'))
        decoder.load_state_dict(torch.load('model/decoder.pth'))
        evaluateRandomly(encoder, decoder)

    else:
        print("Invalid command!")  # Handle invalid commands
