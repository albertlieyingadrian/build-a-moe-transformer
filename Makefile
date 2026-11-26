# make test STAGE=03 [DIR=path/to/your/code]   run one stage's tests against your code (default: the last solution)
# make test-upto STAGE=08 [DIR=...]           run stages 01..STAGE
# make test-solutions                         every reference solution against its own stage, then the last one
#                                              against every stage (what CI runs)
# make solutions                              regenerate solutions/python/01..16 from the last one
# make data                                  download tinyshakespeare (1 MB) for stages 03 and 13-17
# make run-all                                every command behind the numbers in the stage descriptions (3 seeds)
PY ?= python3
LAST := $(lastword $(sort $(wildcard solutions/python/*/code)))
DIR ?= $(LAST)
STAGES := $(shell seq -w 1 17)

test:
	PROGRAM_DIR=$(DIR) PYTHON=$(PY) $(PY) -m pytest -q tests/test_stage_$(STAGE).py

test-upto:
	@files=""; for s in $(STAGES); do [ $$s -le $(STAGE) ] && files="$$files tests/test_stage_$$s.py"; done; \
	PROGRAM_DIR=$(DIR) PYTHON=$(PY) $(PY) -m pytest -q $$files

test-solutions:
	@set -e; for d in solutions/python/*/code; do \
	  n=$$(basename $$(dirname $$d) | cut -c1-2); \
	  echo "== $$d (stage $$n)"; \
	  PROGRAM_DIR=$$d PYTHON=$(PY) $(PY) -m pytest -q tests/test_stage_$$n.py; \
	done
	PROGRAM_DIR=$(LAST) PYTHON=$(PY) $(PY) -m pytest -q tests

solutions:
	$(PY) scripts/make_solutions.py

data:
	mkdir -p data && curl -sSfL -o data/tinyshakespeare.txt \
	  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

run-all: data
	for s in 0 1 2; do PYTHON=$(PY) ./run_all.sh $$s; done

lint:
	$(PY) -m ruff check .

.PHONY: data test test-upto test-solutions solutions run-all lint
