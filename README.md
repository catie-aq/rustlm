RustLM : An efficient Rust CTC Decoder supporting external language models

This an efficient CTC Decoder supporting external language models scorer (Neural models in onnx format, KenLM and dictionnaries as a radix-tree)

This work is initially based on the beautiful work of fast-ctc-decode (https://github.com/nanoporetech/fast-ctc-decode).

## Requirements
```bash
pip install maturin
curl https://sh.rustup.rs -sSf | sh
rustup default nightly
# if maturin is not found later, just add $HOME/.local/bin/ to your PATH
```

## Installation

```bash
git clone https://gitlab.com/sido/asr/rustlm.git --recursive
cd rustlm
make build
pip install target/wheels/rustlm-0.2.0-cp38-cp38-linux_x86_64.whl
```

## TODO List:

- Support batch inference
- Add/improve parallelization/multithreading (rayon?)
- Move probabilities computations into log space (estimate effect on performances)
- Better error management
- Support for dictionnaries boosting
- Support word dictionnaries and kenlm text/binary models
- Support both inference during beam and final inference rescoring (switch needed)
- Improve performances

