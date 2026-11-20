.PHONY: setup test demo serve eval bench docker deploy

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"

test:
	python -m pytest

demo:
	llmgw demo

serve:
	llmgw serve

eval:
	python eval/pii_eval.py

bench:
	python eval/bench.py

docker:
	docker build -t enterprise-llm-gateway . && docker run --rm -p 8080:8080 enterprise-llm-gateway

deploy:        ## needs AWS credentials, Docker and the CDK CLI
	pip install -e ".[infra]" && cd infra && cdk deploy
