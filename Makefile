CONFIG ?= configs/b300.yaml
RESUME ?=
FLAGS  ?= --accept-unverified
RUN    := silentdoubt --config $(CONFIG) $(FLAGS) $(if $(RESUME),--resume $(RESUME))

.PHONY: help install gates rollout labels probes figures report all clean-figures

help:
	@echo "silentdoubt — stages run in order; each is independently resumable."
	@echo ""
	@echo "  make install                  editable install with the judge extra"
	@echo "  make gates                    GPU  ~15 min  loader_contract.gates"
	@echo "  make rollout                  GPU  the turn loop, battery and capture"
	@echo "  make labels                   CPU  behavioural coding + Claude judge"
	@echo "  make probes                   CPU  the full probe suite"
	@echo "  make figures                  CPU  analysis tables + the eight figures"
	@echo "  make report                   CPU  report.md"
	@echo "  make all                      the whole chain"
	@echo ""
	@echo "  CONFIG=configs/b300.yaml      which run config to use"
	@echo "  RESUME=b300                   resume an existing run id"

install:
	pip install -e ".[judge]"

gates:
	$(RUN) gates

rollout:
	$(RUN) rollout

labels:
	$(RUN) labels

probes:
	$(RUN) probes

figures:
	$(RUN) figures

report:
	$(RUN) report

all:
	$(RUN) all

clean-figures:
	rm -f runs/*/figures/*.png runs/*/figures/*.svg
