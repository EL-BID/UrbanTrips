.PHONY: lint unit integration build ci

lint:
	ruff check urbantrips

unit:
	pytest urbantrips/tests/unit -q --cov=urbantrips --cov-report=xml:coverage-unit.xml

integration:
	pytest urbantrips/tests/integration -q --cov=urbantrips --cov-report=xml:coverage-integration.xml

build:
	python -m build

ci: lint unit integration build
