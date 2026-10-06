output "cluster_name" {
  value = module.eks.cluster_name
}
output "kubeconfig_cmd" {
  value = "aws eks update-kubeconfig --region ${var.region} --name ${module.eks.cluster_name}"
}
