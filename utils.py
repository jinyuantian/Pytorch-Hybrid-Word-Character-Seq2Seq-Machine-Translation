#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N: Homework 5
"""

import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def pad_sents_char(sents, char_pad_token):
    """ Pad list of sentences according to the longest sentence in the batch and max_word_length.
    @param sents (list[list[list[int]]]): list of sentences, result of `words2charindices()`
        from `vocab.py`
    @param char_pad_token (int): index of the character-padding token
    @returns sents_padded (list[list[list[int]]]): list of sentences where sentences/words shorter
        than the max length sentence/word are padded out with the appropriate pad token, such that
        each sentence in the batch now has same number of words and each word has an equal
        number of characters
        Output shape: (batch_size, max_sentence_length, max_word_length)
    """
    # Words longer than 21 characters should be truncated
    max_word_length = 21

    # max_sen_length = max(len(s) for s in sents)
    # sents_padded = [[word[:max_word_length-1] + [word[-1]] if len(word) >= max_word_length else word[:] + [char_pad_token] * (max_word_length - len(word)) for word in s] for s in sents]
    # sents_padded = [s + [[char_pad_token] * max_word_length for _ in range(max_sen_length-len(s))] if len(s) < max_sen_length else s for s in sents_padded]

    longest = max(len(sent) for sent in sents)

    sents = list(map(lambda sent: sent + [[]] * (longest - len(sent)), sents))
    sents_trunc = list(list(map(lambda word: word[:max_word_length-1] + [word[-1]] if len(word) > max_word_length else word
                                 , sent)) for sent in sents)
    sents_padded = list(list(map(lambda word: word + [char_pad_token] * (max_word_length - len(word)), sent))
                        for sent in sents_trunc)
    return sents_padded


def pad_sents(sents, pad_token):
    """ Pad list of sentences according to the longest sentence in the batch.
    @param sents (list[list[int]]): list of sentences, where each sentence
                                    is represented as a list of words
    @param pad_token (int): padding token
    @returns sents_padded (list[list[int]]): list of sentences where sentences shorter
        than the max length sentence are padded out with the pad_token, such that
        each sentences in the batch now has equal length.
        Output shape: (batch_size, max_sentence_length)
    """

    max_len = max(len(s) for s in sents)
    sents_padded = list(map(lambda sent: sent + [pad_token]*(max_len-len(sent)), sents))

    return sents_padded



def read_corpus(file_path, source):
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path (str): path to file containing corpus
    @param source (str): "tgt" or "src" indicating whether text
        is of the source language or target language
    """
    data = []
    for line in open(file_path):
        sent = line.strip().split(' ')
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data


def batch_iter(data, batch_size, shuffle=False):
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    @param data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents
