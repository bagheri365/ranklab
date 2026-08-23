.PHONY: setup test lint typecheck audit freeze-m0

setup:
	python -m pip install -e '.[dev]'

test:
	pytest

lint:
	ruff check src tests

typecheck:
	mypy src/ranklab

audit:
	ranklab audit --benchmark configs/benchmark.yaml

freeze-m0:
	ranklab freeze-protocol --protocol research/protocol_frozen_m0.yaml
