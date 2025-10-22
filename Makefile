.PHONY: help setup-inference setup-training build trt-build trt-rebuild up down logs restart clean-cache clean-model clean-all reverse-proxy-config reverse-proxy-setup reverse-proxy-run reverse-proxy-dev

PYTHON ?= python
PIP ?= uv pip
REVERSE_PROXY_CONFIG := $(abspath reverse_proxy/config.json)
REVERSE_PROXY_CONFIG_EXAMPLE := $(abspath reverse_proxy/config.example.json)
REVERSE_PROXY_DEPS_SENTINEL := reverse_proxy/.deps-installed
REVERSE_PROXY_REQUIREMENTS := reverse_proxy/requirements.txt

reverse-proxy-config:
	@if [ -f $(REVERSE_PROXY_CONFIG) ]; then \
		echo "reverse_proxy/config.json already exists (skipping)."; \
	else \
		cp $(REVERSE_PROXY_CONFIG_EXAMPLE) $(REVERSE_PROXY_CONFIG); \
		echo "Generated reverse_proxy/config.json"; \
	fi

reverse-proxy-setup: reverse-proxy-config $(REVERSE_PROXY_DEPS_SENTINEL)


$(REVERSE_PROXY_DEPS_SENTINEL): $(REVERSE_PROXY_REQUIREMENTS)
	$(PIP) install -r $(REVERSE_PROXY_REQUIREMENTS)
	touch $(REVERSE_PROXY_DEPS_SENTINEL)

reverse-proxy-run: reverse-proxy-config $(REVERSE_PROXY_DEPS_SENTINEL)
	REVERSE_PROXY_CONFIG_PATH=$(REVERSE_PROXY_CONFIG) $(PYTHON) -m reverse_proxy.server

reverse-proxy-dev: reverse-proxy-run

help:
	@echo "Dippy Studio Bittensor Miner Commands:"
	@echo ""
	@echo "  🚀 Deployment Modes:"
	@echo "    make setup-inference - Configure and deploy INFERENCE server only"
	@echo "    make setup-training  - Configure and deploy TRAINING server only"
	@echo "    make setup-kontext   - Deploy with FLUX.1-Kontext-dev editing enabled"
	@echo ""
	@echo "  Individual Steps:"
	@echo "    make build       - Build Docker images (uses cache)"
	@echo "    make rebuild     - Force rebuild Docker images (no cache)"
	@echo "    make trt-build   - Build TRT engine in container (skips if exists)"
	@echo "    make trt-rebuild - Force rebuild TRT engine in container"
	@echo "    make up          - Start miner service"
	@echo "    make down        - Stop miner service"
	@echo "    make logs        - Follow miner logs"
	@echo "    make restart     - Restart miner service"
	@echo ""
	@echo "  Reverse Proxy:"
	@echo "    make reverse-proxy-setup - Install deps and prepare config"
	@echo "    make reverse-proxy-run   - Start the reverse proxy"
	@echo "    make reverse-proxy-dev   - Start the proxy (alias of run)"
	@echo "    make reverse-proxy-config - Copy example config if missing"
	@echo ""
	@echo "  Testing:"
	@echo "    make test-kontext-determinism - Run Kontext determinism E2E tests"
	@echo "    make test-kontext-unit        - Run Kontext unit tests"
	@echo ""
	@echo "  Maintenance:"
	@echo "    make clean-cache - Remove all cached TRT engines"
	@echo "    make clean-model - Remove downloaded FLUX model"
	@echo "    make clean-all   - Remove TRT engines and FLUX model"

# Build Docker images
build:
	docker compose build trt-builder miner

# Force rebuild without cache (use sparingly)
rebuild:
	docker compose build --no-cache trt-builder miner

# Build TRT engine in container
trt-build:
	docker compose run --rm trt-builder

# Force rebuild TRT engine
trt-rebuild:
	docker compose run --rm trt-builder --force

# Start miner service
up:
	docker compose up -d miner

# Stop miner service
down:
	docker compose down

# Follow miner logs
logs:
	docker compose logs -f --tail=200 miner

# Restart miner
restart: down up

# Clean TRT cache
clean-cache:
	@echo "WARNING: This will delete all cached TRT engines!"
	@printf "Are you sure? [y/N] "; \
	read REPLY; \
	case "$$REPLY" in \
		[yY]) \
			sudo rm -rf trt-cache/*; \
			echo "TRT cache cleared.";; \
		*) \
			echo "Cancelled.";; \
	esac

# Clean downloaded model
clean-model:
	@echo "WARNING: This will delete the downloaded FLUX.1-dev model!"
	@printf "Are you sure? [y/N] "; \
	read REPLY; \
	case "$$REPLY" in \
		[yY]) \
			sudo rm -rf /models/FLUX.1-dev; \
			echo "Model removed.";; \
		*) \
			echo "Cancelled.";; \
	esac

# Clean everything
clean-all:
	@echo "WARNING: This will delete TRT engines AND the FLUX model!"
	@printf "Are you sure? [y/N] "; \
	read REPLY; \
	case "$$REPLY" in \
		[yY]) \
			sudo rm -rf trt-cache/*; \
			sudo rm -rf /models/FLUX.1-dev; \
			echo "All cleaned.";; \
		*) \
			echo "Cancelled.";; \
	esac

# Setup for INFERENCE mode only
setup-inference:
	@echo "📦 Setting up INFERENCE deployment..."
	@echo ""
	@echo "⚠️  Configuring for inference-only mode"
	@echo ""
	@echo "🔍 Checking base model components..."
	@if [ ! -d "/models/FLUX.1-dev" ]; then \
		echo "⚠️  Base model not found!"; \
		echo "📥 Downloading FLUX.1-dev model components..."; \
		echo "   (Required for tokenizer, VAE, and scheduler)"; \
		echo ""; \
		docker compose run --rm miner huggingface-cli download black-forest-labs/FLUX.1-dev --local-dir /models/FLUX.1-dev; \
		echo "✓ Base model components downloaded"; \
	else \
		echo "✓ Base model components found"; \
	fi
	@echo ""
	@echo "🔍 Checking for TRT engines..."
	@if [ ! -d "trt-cache" ] || [ -z "$$(ls -A trt-cache)" ]; then \
		echo "⚠️  TRT engines not found!"; \
		echo "🔨 Building TRT engines (this will take 20-30 minutes)..."; \
		echo ""; \
		$(MAKE) trt-build; \
	else \
		echo "✓ TRT engines found"; \
	fi
	@echo ""
	@echo "🚀 Starting inference service (training disabled)..."
	ENABLE_TRAINING=false ENABLE_INFERENCE=true docker compose up -d miner
	@echo ""
	@echo "✅ Inference service deployed!"
	@echo "   API: http://localhost:8091"
	@echo "   Logs: make logs"

# Setup for TRAINING mode only
setup-training:
	@echo "📦 Setting up TRAINING deployment..."
	@echo ""
	@echo "⚠️  Configuring for training-only mode"
	@echo ""
	@echo "🔍 Checking base model..."
	@if [ ! -d "/models/FLUX.1-dev" ]; then \
		echo "⚠️  Base model not found!"; \
		echo "📥 Downloading FLUX.1-dev model (this may take a while)..."; \
		echo ""; \
		docker compose run --rm miner huggingface-cli download black-forest-labs/FLUX.1-dev --local-dir /models/FLUX.1-dev; \
		echo "✓ Base model downloaded"; \
	else \
		echo "✓ Base model found"; \
	fi
	@echo ""
	@echo "🚀 Starting training service (inference disabled)..."
	ENABLE_INFERENCE=false ENABLE_TRAINING=true docker compose up -d miner
	@echo ""
	@echo "✅ Training service deployed!"
	@echo "   API: http://localhost:8091"
	@echo "   Logs: make logs"

.PHONY: setup-kontext
setup-kontext:  ## Deploy with FLUX.1-Kontext-dev editing enabled
	@echo "Setting up Kontext editing..."
	@grep -q "ENABLE_KONTEXT_EDIT=true" .env || echo "ENABLE_KONTEXT_EDIT=true" >> .env
	@grep -q "PYTHONHASHSEED=0" .env || echo "PYTHONHASHSEED=0" >> .env
	@grep -q "CUBLAS_WORKSPACE_CONFIG=:4096:8" .env || echo 'CUBLAS_WORKSPACE_CONFIG=:4096:8' >> .env
	@mkdir -p output/edits
	$(MAKE) build
	$(MAKE) up
	@echo "Kontext editing enabled! Visit http://localhost:8091 for API docs"

.PHONY: test-kontext-determinism
test-kontext-determinism:  ## Run Kontext determinism E2E tests
	@echo "Running Kontext determinism tests..."
	docker compose exec miner pytest tests/e2e/kontext_determinism/ -v -s

.PHONY: test-kontext-unit
test-kontext-unit:  ## Run Kontext unit tests
	@echo "Running Kontext unit tests..."
	docker compose exec miner pytest tests/unit/test_kontext_pipeline.py -v
