clean:
	rm -rf *~ dist *.egg-info build target

release:
	maturin build --release --manylinux=off

build:
	maturin build --manylinux=off

develop:
	maturin develop

test: develop
	cargo test
	python3 tests/test_decode.py
