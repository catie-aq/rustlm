#!/usr/env/bin python3

import os.path
import numpy as np
from unittest import TestCase, main
from rustlm import *

class Test1DBeamSearch(TestCase):
    def setUp(self):
        self.beam_width = 32
        self.alphabet = [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
         "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'"]

        self.cutoff_prob = 0.01
        self.probs = self.get_real_data()

    def get_real_data(self):
        feats = np.load("probs.npy")
        return feats

    def test_gpt2_beam_search(self):
        """ gpt based beam search test with the canonical alphabet """
        # assume that file are in the test directory
        if (not os.path.isfile('french_tokenizer-vocab.json')) or (not os.path.isfile('french_tokenizer-merges.txt')) or (not os.path.isfile('model.onnx')):
            print("GPT2 test diasabled - files not found.")
            return

        gpt2_beam_search = GPT2BeamSearch("french_tokenizer-vocab.json", "french_tokenizer-merges.txt", "model.onnx", 1, "<pad>")
        seqs, paths, prbs = gpt2_beam_search.beam_search(self.probs, self.alphabet, self.beam_width, self.cutoff_prob, 0.5, 0.0, len(self.alphabet), self.alphabet.index(" "))
        print(seqs)
        print(paths)
        print(prbs)

    def test_nolm_beam_search(self):
        """ simple beam search test with the canonical alphabet """
        nolm_beam_search = NoLMBeamSearch()
        seqs, paths, prbs = nolm_beam_search.beam_search(self.probs, self.alphabet, self.beam_width, self.cutoff_prob, 0.0, 0.0, len(self.alphabet), self.alphabet.index(" "))
        print(seqs)
        print(paths)
        print(prbs)



if __name__ == '__main__':
    main()
