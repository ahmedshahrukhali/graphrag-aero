.PHONY: help infra up down logs models ingest eval test
help:        ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-10s %s\n",$$1,$$2}'
infra:       ## start backing services
	docker compose up -d qdrant neo4j postgres ollama otel-collector
up:          ## start full stack
	docker compose up --build backend frontend
down:        ## stop everything
	docker compose down
logs:        ## tail logs
	docker compose logs -f
models:      ## pull the Ollama model named in .env
	docker compose exec ollama ollama pull $${OLLAMA_MODEL:-gemma2:9b}
ingest:      ## run ingestion over data/corpus
	docker compose exec backend python -m ingestion.run
eval:        ## run the evaluation harness
	docker compose exec backend python -m eval.run
test:        ## run unit tests (no network/weights)
	docker compose exec backend pytest -q
