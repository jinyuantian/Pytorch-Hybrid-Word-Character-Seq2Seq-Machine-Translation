# Hybrid Word Character Seq2Seq Machine Translation

The project is my solution to the 5th assignment of CS224n (Natural Language Processing with Deep Learning) offered by Stanford University Winter 2019.

Implementation of a Seq2Seq model with attention mechanism that translates German sentences into English. It applies a character-based 1-D convolutional network for word embeddings. It is a hybrid system that translates sentences mostly at word level and resort to a character-based decoder for rare word. The character-level recurrent neural networks compute source word
representations and recover unknown target words when needed. The twofold advantage of such a hybrid approach is that it is much faster and easier to train than character-based ones; at the same time, it never produces unknown words as in the case of word-based models.
