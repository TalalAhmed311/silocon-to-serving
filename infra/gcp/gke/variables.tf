variable "project_id" { type = string }
variable "owner" { type = string }
variable "billing_account" {
  type        = string
  description = "Billing account ID (XXXXXX-XXXXXX-XXXXXX) for the budget alert"
}
variable "budget_email_channel" {
  type        = string
  default     = ""
  description = "Optional Cloud Monitoring notification channel ID; without it the budget emails billing admins"
}
variable "monthly_budget_usd" {
  type    = number
  default = 300
}
variable "region" {
  type    = string
  default = "us-central1"
}
variable "zone" {
  type    = string
  default = "us-central1-a" # pick a zone that offers your GPU type (gcloud compute accelerator-types list)
}
variable "gpu_machine_type" {
  type    = string
  default = "g2-standard-4" # 1x L4 — the closest analogue of AWS g6.xlarge (UNVERIFIED: check GCP machine docs)
}
variable "gpu_type" {
  type    = string
  default = "nvidia-l4"
}
variable "gpu_spot" {
  type    = bool
  default = false
}
variable "gpu_max" {
  type    = number
  default = 2
}
variable "master_authorized_cidrs" {
  type        = list(string)
  description = "CIDRs allowed to reach the GKE control plane (your IP /32). Never the whole internet."
}
