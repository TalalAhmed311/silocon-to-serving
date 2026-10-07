variable "region" {
  type        = string
  default     = "us-east-1"
  description = "Pick the region where your G/P quota was granted and the instance type is offered."
}

variable "owner" {
  type        = string
  description = "Your name or handle; tagged on every resource so you can find (and delete) them."
}

variable "lab" {
  type        = string
  default     = "p2"
  description = "Free-form label for the module using this environment (e.g. p2.1, p5)."
}

variable "instance_type" {
  type        = string
  default     = "g6.xlarge"
  description = "g4dn.xlarge (T4, cheapest CUDA), g5.xlarge (A10G), g6.xlarge (L4, FP8), g6e.xlarge (L40S)."
}

variable "use_spot" {
  type        = bool
  default     = false
  description = "Spot is cheaper but can be reclaimed with 2 minutes' notice. Fine for benches you can rerun."
}

variable "root_volume_gb" {
  type    = number
  default = 200 # DLAMI + CUDA + a container image + one 8B model in bf16 fits comfortably
}

variable "idle_minutes" {
  type        = number
  default     = 30
  description = "Auto-stop after this many consecutive minutes with GPU utilization < 5% and no SSM session."
}

variable "monthly_budget_usd" {
  type    = number
  default = 100
}

variable "budget_email" {
  type        = string
  description = "Where AWS Budgets sends alerts (80% forecast, 100% actual)."
}

variable "weights_bucket" {
  type        = string
  default     = ""
  description = "Optional S3 bucket holding model weights; the instance role gets read-only access to weights_prefix in it."
}

variable "weights_prefix" {
  type    = string
  default = "models/"
}

variable "ami_id" {
  type        = string
  default     = ""
  description = "Override the AMI. Default: newest AWS Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)."
}
