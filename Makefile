clean:
	rm -rf *~ dist *.egg-info build target
	$(MAKE) clean -C submodules/triton-rust/src/shared_memory

release:
	maturin build --release --manylinux=off

dependencies:
	$(MAKE) -C submodules/triton-rust/src/shared_memory

build: dependencies
	maturin build --manylinux=off

develop:
	maturin develop

test: develop
	cargo test
	python3 tests/test_decode.py
