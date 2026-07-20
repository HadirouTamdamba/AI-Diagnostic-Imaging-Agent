output "app_url" {
  description = "Public URL of the deployed application"
  value       = "http://${aws_lb.app.dns_name}"
}

output "ecr_repository_url" {
  description = "Push the image here before the service can start"
  value       = aws_ecr_repository.app.repository_url
}

output "cluster_name" {
  description = "ECS cluster name"
  value       = aws_ecs_cluster.app.name
}

output "log_group" {
  description = "CloudWatch log group for the app"
  value       = aws_cloudwatch_log_group.app.name
}
