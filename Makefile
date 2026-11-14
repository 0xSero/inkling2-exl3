IMAGE ?= ghcr.io/0xsero/inkling2-exl3:latest
MODELS ?= $(HOME)/models
build:
	docker build -t $(IMAGE) .
sweep:
	docker run --rm --gpus all --ipc=host -v $(MODELS):/models $(IMAGE) sweep
serve:
	docker run --rm --gpus all --ipc=host -p 8000:8000 -v $(MODELS):/models $(IMAGE) serve
audit:
	scripts/audit.sh
.PHONY: build sweep serve audit
