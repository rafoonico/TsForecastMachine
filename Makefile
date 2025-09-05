.PHONY: prep train compare

prep:
	python -m src.data_prep.build_dataset  # placeholder

train:
	python -m src.train

compare:
	python -m src.compare
