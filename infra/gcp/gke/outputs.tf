output "get_credentials" {
  value = "gcloud container clusters get-credentials ${google_container_cluster.c.name} --zone ${var.zone} --project ${var.project_id}"
}
output "workload_pool" { value = "${var.project_id}.svc.id.goog" }
