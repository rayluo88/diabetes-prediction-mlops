# Makefile for Diabetes Prediction MLOps Project
# Comprehensive commands for development, testing, and deployment

.PHONY: help setup install install-dev clean test test-unit test-integration \
        lint format check-format type-check security-check \
        run-data-pipeline run-training run-api run-monitoring \
        docker-build docker-run docker-stop docker-clean \
        terraform-plan terraform-apply terraform-destroy \
        deploy-local deploy-cloud ci-check pre-commit-install \
        notebook-clean monitoring-dashboard

# Default target
help: ## Show this help message
	@echo "Diabetes Prediction MLOps Project - Available Commands:"
	@echo "======================================================"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Environment Setup
setup: ## Complete project setup (install + dev tools + pre-commit)
	@echo "🚀 Setting up Diabetes Prediction MLOps environment..."
	$(MAKE) install-dev
	$(MAKE) pre-commit-install
	@echo "✅ Setup complete!"

install: ## Install production dependencies
	@echo "📦 Installing production dependencies..."
	python install_deps.py

install-dev: ## Install development dependencies
	@echo "📦 Installing all dependencies..."
	python install_deps.py --dev

clean: ## Clean up generated files and caches
	@echo "🧹 Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	rm -f bandit-report.json

# Code Quality
lint: ## Run all linting checks
	@echo "🔍 Running linting checks..."
	flake8 src/ tests/
	mypy src/
	bandit -r src/ -f json -o bandit-report.json || true
	yamllint .
	@echo "✅ Linting complete!"

format: ## Format all Python code
	@echo "🎨 Formatting code..."
	black src/ tests/
	isort src/ tests/
	@echo "✅ Code formatted!"

check-format: ## Check code formatting without making changes
	@echo "🔍 Checking code format..."
	black --check src/ tests/
	isort --check-only src/ tests/

type-check: ## Run type checking with mypy
	@echo "🔍 Running type checks..."
	mypy src/

security-check: ## Run security vulnerability checks
	@echo "🔒 Running security checks..."
	bandit -r src/ -f json -o bandit-report.json
	safety check
	@echo "✅ Security checks complete!"

# Testing
test: ## Run all tests with coverage
	@echo "🧪 Running all tests..."
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term
	@echo "✅ All tests complete!"

test-unit: ## Run unit tests only
	@echo "🧪 Running unit tests..."
	pytest tests/unit/ -v

test-integration: ## Run integration tests only
	@echo "🧪 Running integration tests..."
	pytest tests/integration/ -v

# Pipeline Commands
run-data-pipeline: ## Load and preprocess diabetes dataset
	@echo "📊 Running data pipeline..."
	python src/data/load_diabetes_data.py
	python src/data/preprocess.py
	@echo "✅ Data pipeline complete!"

run-training: ## Train diabetes prediction models
	@echo "🤖 Training diabetes models..."
	python src/models/train_diabetes_model.py
	@echo "✅ Model training complete!"

run-api: ## Start FastAPI diabetes prediction service
	@echo "🚀 Starting diabetes prediction API..."
	uvicorn src.api.diabetes_api:app --host 0.0.0.0 --port 8000 --reload

run-monitoring: ## Run model monitoring pipeline
	@echo "📊 Running monitoring pipeline..."
	python src/monitoring/diabetes_monitor.py
	@echo "✅ Monitoring complete!"

# Workflow Commands
run-prefect-training: ## Run Prefect training pipeline
	@echo "🔄 Running Prefect training pipeline..."
	python src/workflows/training_pipeline.py

run-prefect-monitoring: ## Run Prefect monitoring pipeline
	@echo "📊 Running Prefect monitoring pipeline..."
	python src/workflows/monitoring_pipeline.py

run-prefect-server: ## Start Prefect web server
	@echo "🌐 Starting Prefect web server..."
	prefect server start --host 0.0.0.0 --port 4200

run-mlflow-ui: ## Start MLflow UI
	@echo "📊 Starting MLflow UI..."
	mlflow ui --host 0.0.0.0 --port 5000

# Docker Commands
docker-build: ## Build all Docker images
	@echo "🐳 Building Docker images..."
	docker build -f deployment/docker/Dockerfile.api -t diabetes-api:latest .
	docker build -f deployment/docker/Dockerfile.mlflow -t diabetes-mlflow:latest .
	@echo "✅ Docker images built!"

docker-run: ## Run local Docker stack
	@echo "🐳 Starting Docker stack..."
	cd deployment/docker && docker-compose up -d
	@echo "✅ Docker stack running!"
	@echo "🌐 Services available at:"
	@echo "  - API: http://localhost:8000"
	@echo "  - MLflow: http://localhost:5000"
	@echo "  - Prefect: http://localhost:4200"
	@echo "  - Grafana: http://localhost:3000"

docker-stop: ## Stop Docker stack
	@echo "🐳 Stopping Docker stack..."
	cd deployment/docker && docker-compose down

docker-clean: ## Clean up Docker resources
	@echo "🐳 Cleaning Docker resources..."
	cd deployment/docker && docker-compose down -v
	docker system prune -f

# Terraform Commands
terraform-init: ## Initialize Terraform
	@echo "🏗️ Initializing Terraform..."
	cd deployment/terraform && terraform init

terraform-plan: ## Plan Terraform deployment
	@echo "🏗️ Planning Terraform deployment..."
	cd deployment/terraform && terraform plan

terraform-apply: ## Apply Terraform deployment
	@echo "🏗️ Applying Terraform deployment..."
	cd deployment/terraform && terraform apply -auto-approve

terraform-destroy: ## Destroy Terraform infrastructure
	@echo "🏗️ Destroying Terraform infrastructure..."
	cd deployment/terraform && terraform destroy -auto-approve

# Deployment Commands
deploy-local: ## Deploy complete local development stack
	@echo "🚀 Deploying local stack..."
	$(MAKE) run-data-pipeline
	$(MAKE) run-training
	$(MAKE) docker-run
	@echo "✅ Local deployment complete!"

deploy-cloud: ## Deploy to AWS cloud
	@echo "☁️ Deploying to AWS..."
	$(MAKE) terraform-init
	$(MAKE) terraform-apply
	@echo "✅ Cloud deployment complete!"

# CI/CD Commands
ci-check: ## Run all CI checks (formatting, linting, testing)
	@echo "🔄 Running CI checks..."
	$(MAKE) check-format
	$(MAKE) lint
	$(MAKE) type-check
	$(MAKE) security-check
	$(MAKE) test
	@echo "✅ All CI checks passed!"

pre-commit-install: ## Install pre-commit hooks
	@echo "🪝 Installing pre-commit hooks..."
	pre-commit install
	pre-commit install --hook-type commit-msg
	@echo "✅ Pre-commit hooks installed!"

pre-commit-run: ## Run pre-commit hooks on all files
	@echo "🪝 Running pre-commit hooks..."
	pre-commit run --all-files

# Notebook Commands
notebook-start: ## Start Jupyter notebook server
	@echo "📓 Starting Jupyter notebook..."
	jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

notebook-clean: ## Clean notebook outputs
	@echo "📓 Cleaning notebook outputs..."
	nbqa black notebooks/
	jupyter nbconvert --clear-output --inplace notebooks/*.ipynb

# Monitoring Commands
monitoring-dashboard: ## Open monitoring dashboard
	@echo "📊 Opening monitoring dashboard..."
	@echo "🌐 Available dashboards:"
	@echo "  - MLflow: http://localhost:5000"
	@echo "  - Prefect: http://localhost:4200"
	@echo "  - Grafana: http://localhost:3000"
	@echo "  - Evidently: http://localhost:8085"

# Data Commands
download-data: ## Download and prepare datasets
	@echo "📥 Downloading diabetes dataset..."
	$(MAKE) run-data-pipeline

# Model Commands
train-models: ## Train all model variants
	@echo "🤖 Training all models..."
	$(MAKE) run-training

evaluate-models: ## Evaluate trained models
	@echo "📊 Evaluating models..."
	python -c "from src.models.train_diabetes_model import DiabetesModelTrainer; trainer = DiabetesModelTrainer(); print('Model evaluation complete')"

# Health Checks
health-check: ## Check health of all services
	@echo "🏥 Checking service health..."
	@curl -s http://localhost:8000/health || echo "❌ API service down"
	@curl -s http://localhost:5000/health || echo "❌ MLflow service down"
	@curl -s http://localhost:4200/api/health || echo "❌ Prefect service down"

# Development Helpers
dev-setup: ## Quick development environment setup
	@echo "⚡ Quick dev setup..."
	$(MAKE) install-dev
	$(MAKE) run-data-pipeline
	@echo "✅ Development environment ready!"

full-pipeline: ## Run complete ML pipeline
	@echo "🔄 Running complete Prefect pipeline..."
	$(MAKE) run-data-pipeline
	$(MAKE) run-prefect-training
	$(MAKE) run-prefect-monitoring
	@echo "✅ Complete pipeline finished!"

# Project Information
project-info: ## Show project information
	@echo "🏥 Diabetes Prediction MLOps Project"
	@echo "=================================="
	@echo "Purpose: Singapore healthcare diabetes prediction"
	@echo "Tech Stack: Python, MLflow, Prefect, FastAPI, Evidently, Terraform"
	@echo "Cloud: AWS (ECS, RDS, S3)"
	@echo "Monitoring: Comprehensive drift detection and bias analysis"
	@echo ""
	@echo "📁 Project Structure:"
	@echo "  src/          - Source code (data, models, api, monitoring)"
	@echo "  tests/        - Unit and integration tests"
	@echo "  deployment/   - Docker and Terraform configurations"
	@echo "  notebooks/    - Jupyter notebooks for exploration"
	@echo ""
	@echo "🚀 Quick Start: make setup && make deploy-local"

# Environment Variables Check
check-env: ## Check required environment variables
	@echo "🔍 Checking environment variables..."
	@python -c "import os; missing = [var for var in ['AWS_REGION'] if not os.getenv(var)]; print('✅ All required env vars set!') if not missing else print(f'❌ Missing: {missing}')"