.PHONY: setup test lint typecheck audit audit-data freeze-m0

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

audit-data:
	ranklab audit-data --data-dir data/raw/KuaiRand-Pure/data --output runs/m0/data_audit.json

freeze-m0:
	ranklab freeze-protocol --protocol research/protocol_frozen_m0.yaml
