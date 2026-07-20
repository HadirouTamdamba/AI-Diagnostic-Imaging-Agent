variable "aws_region" {
  description = "AWS region to deploy into"
  type        = string
  default     = "eu-west-3" # Paris
}

variable "app_name" {
  description = "Base name for all resources"
  type        = string
  default     = "medical-imaging-agent"
}

variable "container_image" {
  description = "Full image URI (ECR). If empty, use the ECR repo created here at :latest."
  type        = string
  default     = ""
}

variable "container_port" {
  description = "Port the Streamlit app listens on"
  type        = number
  default     = 8501
}

variable "health_check_path" {
  description = "ALB health check path"
  type        = string
  default     = "/_stcore/health"
}

variable "cpu" {
  description = "Fargate task CPU units (1024 = 1 vCPU)"
  type        = number
  default     = 1024
}

variable "memory" {
  description = "Fargate task memory in MiB"
  type        = number
  default     = 2048
}

variable "desired_count" {
  description = "Number of running tasks"
  type        = number
  default     = 1
}

variable "google_api_key" {
  description = "Gemini API key, stored in Secrets Manager (never commit this)"
  type        = string
  sensitive   = true
}

variable "model_id" {
  description = "Gemini model id"
  type        = string
  default     = "gemini-flash-latest"
}

variable "enable_web_search" {
  description = "Enable live web-search tool"
  type        = bool
  default     = true
}

variable "default_language" {
  description = "Default UI/report language (en|fr)"
  type        = string
  default     = "en"
}

variable "log_level" {
  description = "Application log level"
  type        = string
  default     = "INFO"
}
