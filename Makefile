.PHONY: install lint format test test-cov build up down clean help

# ── Variables ──────────────────────────────────────────────────────────────────
PYTHON      := python3
PIP         := pip
DOCKER      := docker
COMPOSE     := docker compose
IMAGE_NAME  := stochvol-api
IMAGE_TAG   := latest

# ── Default target ─────────────────────────────────────────────────────────────
help:  ## Show this help message
	@echo "Usage: make <target>"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Python environment ─────────────────────────────────────────────────────────
install:  ## Install Python dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-cov flake8 black isort

# ── Code quality ───────────────────────────────────────────────────────────────
lint:  ## Run flake8 linter
	flake8 src/ app/ --max-line-length=120 --ignore=E203,W503

format:  ## Auto-format code with black and isort
	black src/ app/ tests/
	isort src/ app/ tests/

format-check:  ## Check formatting without modifying files
	black --check src/ app/ tests/
	isort --check-only src/ app/ tests/

# ── Testing ────────────────────────────────────────────────────────────────────
test:  ## Run unit tests
	pytest tests/ -v

test-cov:  ## Run tests with coverage report
	pytest tests/ -v \
	  --cov=src \
	  --cov=app \
	  --cov-report=term-missing \
	  --cov-report=html:htmlcov

test-fast:  ## Run tests skipping slow ones
	pytest tests/ -v -m "not slow"

# ── ML pipeline ────────────────────────────────────────────────────────────────
ingest:  ## Fetch and cache market data
	$(PYTHON) -m src.ingestion.fetch_market_data

features:  ## Build feature matrix
	$(PYTHON) -m src.features.build_features

labels:  ## Compute HMM regime labels
	$(PYTHON) -m src.labeling.build_regimes

train:  ## Train all models
	$(PYTHON) -m src.training.train_ml

pipeline: ingest features labels train  ## Run full ML pipeline end-to-end

# ── Docker ─────────────────────────────────────────────────────────────────────
build:  ## Build Docker image
	$(DOCKER) build -t $(IMAGE_NAME):$(IMAGE_TAG) .

up:  ## Start all services with docker compose
	$(COMPOSE) up -d

down:  ## Stop all services
	$(COMPOSE) down

logs:  ## Tail docker compose logs
	$(COMPOSE) logs -f

shell:  ## Open shell in running API container
	$(DOCKER) exec -it stochvol_api /bin/bash

# ── Cleanup ────────────────────────────────────────────────────────────────────
clean:  ## Remove generated artefacts
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage coverage.xml
	$(DOCKER) image rm -f $(IMAGE_NAME):$(IMAGE_TAG) 2>/dev/null || true

clean-data:  ## Remove cached data files
	rm -rf data/ mlruns/ mlflow.db
