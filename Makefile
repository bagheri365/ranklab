.PHONY: setup test lint typecheck audit audit-data audit-regimes audit-support audit-targets audit-training audit-training-contract freeze-m0

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

audit-regimes:
	ranklab audit-regimes --data-dir data/raw/KuaiRand-Pure/data --output runs/m0/regime_audit.json

freeze-m0:
	ranklab freeze-protocol --protocol research/protocol_frozen_m0.yaml

audit-support:
	ranklab audit-support --data-dir data/raw/KuaiRand-Pure/data --output runs/m0/support_audit.json


audit-targets:
	ranklab audit-targets --data-dir data/raw/KuaiRand-Pure/data --output runs/m0/target_audit.json


audit-training:
	ranklab audit-training --data-dir data/raw/KuaiRand-Pure/data --output runs/m0/training_audit.json


audit-training-contract:
	ranklab audit-training-contract --data-dir data/raw/KuaiRand-Pure/data --output runs/m0/training_contract_audit.json
