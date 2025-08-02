"""
Terraform Variables for Diabetes Prediction MLOps Infrastructure
"""

variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "ap-southeast-1" # Singapore region
}

variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "diabetes-prediction"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
  
  validation {
    condition     = contains(["development", "staging", "production"], var.environment)
    error_message = "Environment must be development, staging, or production."
  }
}

# Networking
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24"]
}

variable "private_subnet_cidrs" {
  description = "CIDR blocks for private subnets"
  type        = list(string)
  default     = ["10.0.3.0/24", "10.0.4.0/24"]
}

# Database
variable "rds_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.micro"
}

variable "rds_allocated_storage" {
  description = "RDS allocated storage in GB"
  type        = number
  default     = 20
}

variable "db_username" {
  description = "Database username"
  type        = string
  default     = "mlflow"
}

variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
  default     = "mlflow-password-change-in-production"
}

# ECS Configuration
variable "fargate_cpu" {
  description = "Fargate instance CPU units"
  type        = number
  default     = 256
}

variable "fargate_memory" {
  description = "Fargate instance memory in MB"
  type        = number
  default     = 512
}

variable "api_image_uri" {
  description = "Docker image URI for the API"
  type        = string
  default     = "diabetes-prediction/api:latest"
}

variable "api_desired_count" {
  description = "Desired number of API containers"
  type        = number
  default     = 2
}

variable "api_min_capacity" {
  description = "Minimum number of API containers"
  type        = number
  default     = 1
}

variable "api_max_capacity" {
  description = "Maximum number of API containers"
  type        = number
  default     = 10
}

# Monitoring and Alerting
variable "alert_emails" {
  description = "List of email addresses for alerts"
  type        = list(string)
  default     = ["data-science-team@singapore-health.gov.sg"]
}

# Tags
variable "additional_tags" {
  description = "Additional tags to apply to resources"
  type        = map(string)
  default     = {}
}