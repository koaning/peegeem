build:
	python nbs/build.py

clean:
	rm -rf nbs/__pycache__

work:
	uv run marimo edit nbs/__init__.py

install:
	uv pip install -e . pytest marimo

pypi: check
	uv build
	uv publish

check: build
	pytest nbs/__init__.py
