"""
Terraform Outputs for Diabetes Prediction MLOps Infrastructure
"""

output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.diabetes_vpc.id
}

output "public_subnet_ids" {
  description = "IDs of the public subnets"
  value       = aws_subnet.public_subnets[*].id
}

output "private_subnet_ids" {
  description = "IDs of the private subnets"
  value       = aws_subnet.private_subnets[*].id
}

output "load_balancer_dns_name" {
  description = "DNS name of the load balancer"
  value       = aws_lb.diabetes_alb.dns_name
}

output "load_balancer_zone_id" {
  description = "Zone ID of the load balancer"
  value       = aws_lb.diabetes_alb.zone_id
}

output "api_endpoint" {
  description = "API endpoint URL"
  value       = "http://${aws_lb.diabetes_alb.dns_name}"
}

output "mlflow_tracking_uri" {
  description = "MLflow tracking URI"
  value       = "postgresql://${var.db_username}:${var.db_password}@${aws_db_instance.mlflow_db.endpoint}/mlflow"
  sensitive   = true
}

output "rds_endpoint" {
  description = "RDS instance endpoint"
  value       = aws_db_instance.mlflow_db.endpoint
  sensitive   = true
}

output "s3_bucket_data_lake" {
  description = "S3 bucket name for data lake"
  value       = aws_s3_bucket.diabetes_data_lake.bucket
}

output "s3_bucket_models" {
  description = "S3 bucket name for models"
  value       = aws_s3_bucket.diabetes_models.bucket
}

output "s3_bucket_monitoring" {
  description = "S3 bucket name for monitoring"
  value       = aws_s3_bucket.diabetes_monitoring.bucket
}

output "ecs_cluster_name" {
  description = "Name of the ECS cluster"
  value       = aws_ecs_cluster.diabetes_cluster.name
}

output "ecs_service_name" {
  description = "Name of the ECS service"
  value       = aws_ecs_service.diabetes_api_service.name
}

output "cloudwatch_log_group" {
  description = "CloudWatch log group name"
  value       = aws_cloudwatch_log_group.ecs_logs.name
}

output "ecr_repository_url" {
  description = "ECR repository URL"
  value       = aws_ecr_repository.diabetes_api.repository_url
}

output "sns_topic_arn" {
  description = "SNS topic ARN for alerts"
  value       = aws_sns_topic.diabetes_alerts.arn
}

output "cloudwatch_dashboard_url" {
  description = "CloudWatch dashboard URL"
  value       = "https://${var.aws_region}.console.aws.amazon.com/cloudwatch/home?region=${var.aws_region}#dashboards:name=${aws_cloudwatch_dashboard.diabetes_dashboard.dashboard_name}"
}

# Infrastructure Summary
output "infrastructure_summary" {
  description = "Summary of deployed infrastructure"
  value = {
    environment         = var.environment
    region             = var.aws_region
    api_endpoint       = "http://${aws_lb.diabetes_alb.dns_name}"
    api_health_check   = "http://${aws_lb.diabetes_alb.dns_name}/health"
    api_docs          = "http://${aws_lb.diabetes_alb.dns_name}/docs"
    ecs_cluster       = aws_ecs_cluster.diabetes_cluster.name
    rds_instance      = aws_db_instance.mlflow_db.identifier
    s3_buckets = {
      data_lake  = aws_s3_bucket.diabetes_data_lake.bucket
      models     = aws_s3_bucket.diabetes_models.bucket
      monitoring = aws_s3_bucket.diabetes_monitoring.bucket
    }
    monitoring = {
      cloudwatch_dashboard = aws_cloudwatch_dashboard.diabetes_dashboard.dashboard_name
      sns_alerts          = aws_sns_topic.diabetes_alerts.name
      log_group          = aws_cloudwatch_log_group.ecs_logs.name
    }
  }
}