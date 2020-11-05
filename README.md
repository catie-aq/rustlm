RustLM : An efficient Rust CTC Decoder supporting external language models

This an efficient CTC Decoder supporting external language models scorer (Neural models in onnx format, KenLM and dictionnaries as a radix-tree)

This work is initially based on the beautiful work of fast-ctc-decode (https://github.com/nanoporetech/fast-ctc-decode).

TODO list:

- Support batch inference
- Add/improve parallelization/multithreading (rayon?)
- Move probabilities computations into log space (estimate effect on performances)
- Better error management
- Support for dictionnaries boosting
- Support word dictionnaries and kenlm text/binary models
- Support both inference during beam and final inference rescoring (switch needed)
- Improve performances
