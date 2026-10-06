# EKS cluster for P3/P4: a small system node group + a GPU node group (tainted, scale-from-zero).
# Modules pinned to exact versions. The input names follow terraform-aws-eks v21 as the author understands them:
# UNVERIFIED — run `terraform init && terraform validate` (tools/ci_terraform.sh) and fix names against the
# module's README at v21.26.0 if validation complains.

locals {
  name = "s2s-${var.owner}"
  azs  = slice(data.aws_availability_zones.this.names, 0, 2)
}

data "aws_availability_zones" "this" {
  state = "available"
}

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "6.0.1" # UNVERIFIED pin: check the registry for the version compatible with AWS provider v6

  name            = local.name
  cidr            = "10.42.0.0/16"
  azs             = local.azs
  private_subnets = ["10.42.0.0/19", "10.42.32.0/19"]
  public_subnets  = ["10.42.96.0/22", "10.42.100.0/22"]

  enable_nat_gateway = true
  single_nat_gateway = true # one NAT for a lab cluster: cheaper, not HA (say so in your notes)

  private_subnet_tags = { "kubernetes.io/role/internal-elb" = 1 }
  public_subnet_tags  = { "kubernetes.io/role/elb" = 1 }
}

# Weight pulls from S3 go through a gateway endpoint instead of the NAT (cheaper, faster). P3.2 exercise 2 explains why.
resource "aws_vpc_endpoint" "s3" {
  vpc_id            = module.vpc.vpc_id
  service_name      = "com.amazonaws.${var.region}.s3"
  vpc_endpoint_type = "Gateway"
  route_table_ids   = module.vpc.private_route_table_ids
}

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "21.26.0"

  name               = local.name
  kubernetes_version = var.kubernetes_version
  vpc_id             = module.vpc.vpc_id
  subnet_ids         = module.vpc.private_subnets

  endpoint_public_access       = true
  endpoint_public_access_cidrs = var.allowed_api_cidrs # API reachable only from your IP
  enable_cluster_creator_admin_permissions = true

  addons = {
    coredns                = {}
    kube-proxy             = {}
    vpc-cni                = { before_compute = true }
    eks-pod-identity-agent = { before_compute = true }
  }

  eks_managed_node_groups = {
    system = {
      instance_types = ["m6i.large"]
      min_size       = 1
      max_size       = 2
      desired_size   = 1
      labels         = { "s2s/pool" = "system" }
    }
    gpu = {
      # EKS-optimized AL2023 NVIDIA AMI: drivers + NVIDIA container toolkit preinstalled, so the GPU Operator runs
      # with driver and toolkit disabled (platform/deploy/gpu-operator-values.yaml).
      ami_type       = "AL2023_x86_64_NVIDIA"
      instance_types = var.gpu_instance_types
      capacity_type  = var.gpu_capacity_type
      min_size       = var.gpu_min
      max_size       = var.gpu_max
      desired_size   = var.gpu_min
      disk_size      = 200
      labels         = { "s2s/pool" = "gpu" }
      taints = {
        gpu = { key = "nvidia.com/gpu", value = "present", effect = "NO_SCHEDULE" } # only GPU pods land here
      }
    }
  }
}

module "budget" {
  source    = "../guardrails"
  name      = local.name
  limit_usd = var.monthly_budget_usd
  email     = var.budget_email
}
