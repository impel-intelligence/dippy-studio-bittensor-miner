.PHONY: help setup-inference setup-training build trt-build trt-rebuild up down logs restart clean-cache clean-model clean-all

help:
	@echo "Dippy Studio Bittensor Miner Commands:"
	@echo ""
	@echo "  🚀 Deployment Modes:"
	@echo "    make setup-inference - Configure and deploy INFERENCE server only"
	@echo "    make setup-training  - Configure and deploy TRAINING server only"
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

