.PHONY: help build trt-build trt-rebuild up down logs restart clean-cache full-setup

help:
	@echo "TensorRT Miner Commands:"
	@echo ""
	@echo "  Quick Start:"
	@echo "    make full-setup  - Build images, create TRT engine, and start miner"
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
	@read -p "Are you sure? [y/N] " -n 1 -r; echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf trt-cache/*; \
		echo "Cache cleared."; \
	else \
		echo "Cancelled."; \
	fi

# Full setup: build images, create engine, start miner
full-setup: build trt-build up
	@echo ""
	@echo "✅ Setup complete! Miner is running."
	@echo "   Use 'make logs' to view output."