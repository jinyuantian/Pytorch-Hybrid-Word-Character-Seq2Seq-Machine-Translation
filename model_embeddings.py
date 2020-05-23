#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N: Homework 5
model_embeddings.py: Embeddings for the NMT model
"""
import torch
import torch.nn as nn
from cnn import CNN
from highway import Highway


class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object.
        """
        super(ModelEmbeddings, self).__init__()

        self.e_char = 50
        self.embed_size = embed_size
        self.vocab = vocab
        self.embeddings = nn.Embedding(len(vocab.char2id), self.e_char, padding_idx=0)
        self.cnn = CNN(e_char=self.e_char, e_word=self.embed_size)
        self.highway = Highway(self.embed_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, input: torch.Tensor):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        seq_len, batch_len, word_len = input.shape
        x_reshaped = self.embeddings(input).permute(0, 1, 3, 2)  # after permutation shape: (sentence_length, batch_size, e_char ,max_word_length)
        x_conv_out = self.cnn(x_reshaped.view(-1, self.e_char, word_len))  # (sentence_length * batch_size, e_word)
        x_highway = self.highway(x_conv_out).view(seq_len, batch_len, self.embed_size)
        x_word_embed = self.dropout(x_highway)
        return x_word_embed









