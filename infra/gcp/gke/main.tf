# P3.9: a GKE cluster that mirrors infra/aws/eks — private nodes, a small system pool, a min-0 L4 GPU pool
# (GKE taints GPU nodes with nvidia.com/gpu=present:NoSchedule automatically), Workload Identity instead of IRSA,
# Cloud NAT for egress, and a billing budget. No public IPs on nodes; the control plane accepts only your CIDRs.

resource "google_compute_network" "vpc" {
  name                    = "s2s-${var.owner}"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "nodes" {
  name                     = "s2s-nodes"
  network                  = google_compute_network.vpc.id
  region                   = var.region
  ip_cidr_range            = "10.10.0.0/20"
  private_ip_google_access = true # reach GCS / Artifact Registry without NAT (the S3 gateway endpoint analogue)
  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = "10.20.0.0/16"
  }
  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = "10.30.0.0/20"
  }
}

resource "google_compute_router" "r" {
  name    = "s2s-router"
  network = google_compute_network.vpc.id
  region  = var.region
}

resource "google_compute_router_nat" "nat" {
  name                               = "s2s-nat"
  router                             = google_compute_router.r.name
  region                             = var.region
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"
}

resource "google_container_cluster" "c" {
  name                     = "s2s-${var.owner}"
  location                 = var.zone # zonal: one free-tier-eligible control plane, and GPUs are zonal anyway
  network                  = google_compute_network.vpc.id
  subnetwork               = google_compute_subnetwork.nodes.id
  remove_default_node_pool = true
  initial_node_count       = 1
  deletion_protection      = false # a lab cluster: `make down` must work

  release_channel { channel = "REGULAR" }
  workload_identity_config { workload_pool = "${var.project_id}.svc.id.goog" }
  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }
  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
    master_ipv4_cidr_block  = "172.16.0.0/28"
  }
  master_authorized_networks_config {
    dynamic "cidr_blocks" {
      for_each = var.master_authorized_cidrs
      content { cidr_block = cidr_blocks.value }
    }
  }
  network_policy { enabled = false }
  datapath_provider = "ADVANCED_DATAPATH" # Dataplane V2 enforces NetworkPolicy (P3.8) — UNVERIFIED interaction with network_policy block
}

resource "google_service_account" "nodes" {
  account_id   = "s2s-nodes-${var.owner}"
  display_name = "s2s GKE nodes (least privilege: logging, monitoring, pulling images)"
}

resource "google_project_iam_member" "nodes" {
  for_each = toset(["roles/logging.logWriter", "roles/monitoring.metricWriter", "roles/artifactregistry.reader"])
  project  = var.project_id
  role     = each.value
  member   = "serviceAccount:${google_service_account.nodes.email}"
}

resource "google_container_node_pool" "system" {
  name       = "system"
  cluster    = google_container_cluster.c.id
  node_count = 2
  node_config {
    machine_type    = "e2-standard-4"
    service_account = google_service_account.nodes.email
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]
    workload_metadata_config { mode = "GKE_METADATA" } # pods can't read the node's credentials (the IMDS-hop-limit analogue)
    shielded_instance_config { enable_secure_boot = true }
  }
}

resource "google_container_node_pool" "gpu" {
  name    = "gpu"
  cluster = google_container_cluster.c.id
  autoscaling {
    min_node_count = 0
    max_node_count = var.gpu_max
  }
  node_config {
    machine_type    = var.gpu_machine_type
    spot            = var.gpu_spot
    service_account = google_service_account.nodes.email
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]
    labels          = { "s2s/pool" = "gpu" }
    guest_accelerator {
      type  = var.gpu_type
      count = 1
      gpu_driver_installation_config { gpu_driver_version = "LATEST" } # GKE-managed driver: no GPU Operator needed (differs from EKS)
    }
    workload_metadata_config { mode = "GKE_METADATA" }
    shielded_instance_config { enable_secure_boot = true }
  }
}

resource "google_billing_budget" "b" {
  billing_account = var.billing_account
  display_name    = "s2s-${var.owner}"
  budget_filter { projects = ["projects/${var.project_id}"] }
  amount {
    specified_amount {
      currency_code = "USD"
      units         = tostring(var.monthly_budget_usd)
    }
  }
  dynamic "threshold_rules" {
    for_each = [0.5, 0.8, 1.0]
    content { threshold_percent = threshold_rules.value }
  }
  dynamic "all_updates_rule" {
    for_each = var.budget_email_channel == "" ? [] : [1]
    content { monitoring_notification_channels = [var.budget_email_channel] }
  }
}
