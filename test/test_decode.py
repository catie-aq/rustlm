#!/usr/env/bin python3

import numpy as np
from unittest import TestCase, main
from rustlm import *

class Test1DBeamSearch(TestCase):
    def setUp(self):
        self.beam_width = 64
        self.alphabet = [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
         "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'"]

        self.cutoff_prob = 0.01
        self.probs = self.get_real_data()

    def get_real_data(self):
        feats = np.load("probs.npy")
        return feats

    def test_beam_search(self):
        """ simple beam search test with the canonical alphabet"""
        gpt2_beam_search = GPT2BeamSearch("../../test_rust_inference2/french_tokenizer-vocab.json", "../../test_rust_inference2/french_tokenizer-merges.txt", "../../test_rust_inference2/model.onnx", 1, "<pad>")
        seqs, paths, prbs = gpt2_beam_search.beam_search(self.probs, self.alphabet, self.beam_width, self.cutoff_prob, 2.0, 1.5, len(self.alphabet), self.alphabet.index(" "))
        print(seqs)
        print(paths)
        print(prbs)



if __name__ == '__main__':
    main()
