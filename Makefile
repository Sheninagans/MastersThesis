 install:
	pip install -r requirements.txt

test:
	pytest

run:
	python thesis_code/main.py

notebook:
	jupyter notebook
