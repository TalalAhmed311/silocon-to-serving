# GKE lab cluster. Same verbs as infra/aws/eks.
.PHONY: plan up creds addons down check
plan:  ; terraform init -input=false && terraform plan
up:    ; terraform init -input=false && terraform apply
creds: ; $$(terraform output -raw get_credentials)
addons:
	# no GPU Operator: GKE installs the driver (gpu_driver_installation_config) and its own device plugin
	helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
	helm repo add kedacore https://kedacore.github.io/charts && helm repo update
	helm upgrade --install kps prometheus-community/kube-prometheus-stack -n monitoring --create-namespace \
	  --version "$${KPS_CHART_VERSION:-75.0.0}" -f ../../../platform/observability/kps-values.yaml
	helm upgrade --install keda kedacore/keda -n keda --create-namespace --version "$${KEDA_CHART_VERSION:-2.21.0}"
down:  ; terraform destroy
check:
	@grep -q '0.0.0.0/0' terraform.tfvars && { echo "master_authorized_cidrs must not be the whole internet"; exit 1; } || echo "ok"
