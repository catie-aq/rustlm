#!/usr/env/bin python3

import os.path
import numpy as np
from unittest import TestCase, main
from rustlm import *


TRITON_URI="http://localhost:7001"
DECODER_MACARENA="decoder_macarena"

class Test1DBeamSearch(TestCase):
    def setUp(self):
        self.beam_width = 16

        self.alphabet = [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
         "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'"]

        self.rnnt_alphabet = []
        for line in open('fr-wpe-512.txt', 'r').readlines():
            self.rnnt_alphabet.append(line.strip())

        self.cutoff_prob = 0.01
        self.probs = self.get_real_ctc_data()
        self.rnnt_encoded = self.get_real_rnnt_data()

    def get_real_ctc_data(self):
        feats = np.load("probs_ctc.npy")
        return feats

    def get_real_rnnt_data(self):
        feats = np.load("encoded_rnnt.npy")
        return feats

    def test_gpt_ctc_beam_search(self):
        """ gpt based beam search test with the canonical alphabet """
        print("GPT2 Test.")
        gpt2_beam_search = BeamSearchCTCCausalLMRescoring(TRITON_URI, "gpt-fr-cased-small", 16, 512, 50000);
        seqs, paths, prbs = gpt2_beam_search.beam_search(self.probs, self.alphabet, "characters", self.beam_width, self.cutoff_prob, 0.5, 0.0, len(self.alphabet), self.alphabet.index(" "), 0)
        print(seqs)
        print(prbs)

    def test_nolm_ctc_beam_search(self):
        """ simple beam search test with the canonical alphabet """
        print("NoLM Test.")
        nolm_beam_search = BeamSearchCTCNoLM()
        seqs, paths, prbs = nolm_beam_search.beam_search(self.probs, self.alphabet, "characters", self.beam_width, self.cutoff_prob, 0.0, 0.0, len(self.alphabet), self.alphabet.index(" "), 0)
        print(seqs)
        print(prbs)

    def test_dico_based_ctc_beam_search(self):
        """ simple beam search test with the canonical alphabet """
        if not os.path.isfile('dico.txt'):
            print("Dico test diasabled - file not found.")
            return

        print("Dico Test.")
        dico_beam_search = BeamSearchCTCDico("dico.txt")
        seqs, paths, prbs = dico_beam_search.beam_search(self.probs, self.alphabet, "characters", 1, self.cutoff_prob, 0.0, 0.0, len(self.alphabet), self.alphabet.index(" "), 0)
        print(seqs)
        print(prbs)

    def test_nolm_rnnt_beam_search(self):
        """ simple beam search test with the canonical alphabet """
        print("NoLM RNNT Test.")
        nolm_beam_search_rnnt = BeamSearchRNNTNoLM(TRITON_URI, DECODER_MACARENA, 1, 640)
        seqs, paths, prbs = nolm_beam_search_rnnt.beam_search(self.rnnt_encoded, self.rnnt_alphabet, "wordpiece", 6, self.cutoff_prob, 512, self.rnnt_alphabet.index("[SEP]"), self.rnnt_alphabet.index("[CLS]"))
        print(seqs)
        print(prbs)

    def test_gpt_rnnt_beam_search(self):
        """ simple beam search test with the canonical alphabet """
        print("GPT2 RNNT Test.")

        gpt2_beam_search_rnnt = BeamSearchRNNTCausalLMRescoring(TRITON_URI, DECODER_MACARENA, 1, 640, "gpt-fr-cased-small",
                                                           16, 512, 50000)
        seqs, paths, prbs = gpt2_beam_search_rnnt.beam_search(self.rnnt_encoded, self.rnnt_alphabet, "wordpiece", 6, self.cutoff_prob, 512, self.rnnt_alphabet.index("[SEP]"), self.rnnt_alphabet.index("[CLS]"))
        print(seqs)
        print(prbs)


if __name__ == '__main__':
    main()
