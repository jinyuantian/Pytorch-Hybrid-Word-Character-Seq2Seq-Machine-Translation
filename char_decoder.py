#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N: Homework 5
"""

import torch
import torch.nn as nn


class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """

        super(CharDecoder, self).__init__()

        self.target_vocab = target_vocab
        self.decoderCharEmb = nn.Embedding(len(target_vocab.char2id), char_embedding_size, padding_idx=target_vocab.char2id['<pad>'])
        self.charDecoder = nn.LSTM(char_embedding_size, hidden_size)
        self.char_output_projection = nn.Linear(hidden_size, len(target_vocab.char2id))

    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s_t in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        input_embed = self.decoderCharEmb(input)
        out, dec_hidden = self.charDecoder(input_embed, dec_hidden)
        scores = self.char_output_projection(out)

        return scores, dec_hidden

    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch.
        """

        loss = nn.CrossEntropyLoss(ignore_index=self.target_vocab.char2id['<pad>'], reduction='sum')
        scores, _ = self.forward(char_sequence[:-1], dec_hidden)
        target = char_sequence[1:]
        return loss(scores.view(-1, len(self.target_vocab.char2id)), target.contiguous().view(-1))  # sum of negative loss likelihood

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """
        
        batch_size = initialStates[0].shape[1]
        decodedWords = [''] * batch_size
        dec_hidden = initialStates
        cur_char = torch.tensor([self.target_vocab.char2id['{']] * batch_size, device=device).unsqueeze(0)  # 1 * batch_size

        for t in range(max_length):
            scores, dec_hidden = self.forward(cur_char, dec_hidden)  # 1 * bz * vocab
            cur_char = torch.argmax(scores.squeeze(0), dim=-1)  # shape: (batch,)

            for i in range(batch_size):
                decodedWords[i] += self.target_vocab.id2char[cur_char[i].item()]

            cur_char = cur_char.unsqueeze(0)

        for i in range(batch_size):
            decodedWords[i] = decodedWords[i].partition('}')[0]

        return decodedWords







