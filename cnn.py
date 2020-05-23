#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N: Homework 5
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, e_char, e_word, kernel_size=5):
        super(CNN, self).__init__()
        self.conv_layer = nn.Conv1d(e_char, e_word, kernel_size=kernel_size)

    def forward(self, x_reshaped: torch.Tensor):
        """

        :param x_reshaped: shape (batch_size, e_char, m_word)
        :return: tensor with shape (batch_size, word_embed_size)
        """
        return torch.max(F.relu(self.conv_layer(x_reshaped)), dim=-1)[0]


def main():
    cnn = CNN(50, 40)
    x = torch.randn(100, 50, 8)
    print(cnn(x).shape)


if __name__ == '__main__':
    main()