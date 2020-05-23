#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Highway(nn.Module):
    def __init__(self, word_size: int):
        super(Highway, self).__init__()

        self.proj_layer = nn.Linear(word_size, word_size, bias=True)
        self.gate_layer = nn.Linear(word_size, word_size, bias=True)

    def forward(self, x: torch.Tensor):
        """
        :param x: tensor in which last dimension has size = word_size
        :return: tensor that has the same shape as x
        """
        x_proj = F.relu(self.proj_layer(x))
        x_gate = torch.sigmoid(self.gate_layer(x))

        return x_gate * x_proj + (1 - x_gate) * x


def main():
    word_size = 50
    highway = Highway(word_size)

    x = torch.randn(10,10, 2, word_size)
    print(highway(x).shape)


if __name__ == '__main__':
    main()



