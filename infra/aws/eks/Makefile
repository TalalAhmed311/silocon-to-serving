TF ?= terraform
VARS ?= terraform.tfvars
.PHONY: init up down kubeconfig addons check
init:
	$(TF) init
up: init
	$(TF) apply -var-file=$(VARS)
kubeconfig:
	@$$($(TF) output -raw kubeconfig_cmd)
addons: kubeconfig
	./addons.sh
# Delete in-cluster LoadBalancers/PVCs first or `terraform destroy` can hang on dependent AWS resources (ENIs, ELBs).
down:
	-kubectl delete svc --all-namespaces --field-selector spec.type=LoadBalancer --wait=true
	-helm uninstall kps -n monitoring; helm uninstall keda -n keda; helm uninstall gpu-operator -n gpu-operator
	$(TF) destroy -var-file=$(VARS)
check:
	$(TF) fmt -check && $(TF) init -backend=false -input=false >/dev/null && $(TF) validate
	@! grep -En '0\.0\.0\.0/0' variables.tf terraform.tfvars.example || (echo "0.0.0.0/0 is not allowed for the API endpoint" && exit 1)
