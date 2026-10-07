# Single-node T2 environment. Every lab: `make up`, work, `make down`. See README.md.
TF ?= terraform
VARS ?= terraform.tfvars

.PHONY: init up down ssm forward stop start cost check
init:
	$(TF) init
up: init
	$(TF) apply -var-file=$(VARS)
down:
	$(TF) destroy -var-file=$(VARS)
ssm:
	@$$($(TF) output -raw ssm_shell)
forward:
	@$$($(TF) output -raw port_forward_8000)
stop:
	aws ec2 stop-instances --instance-ids $$($(TF) output -raw instance_id)
start:
	aws ec2 start-instances --instance-ids $$($(TF) output -raw instance_id)
cost:
	@../scripts/cost.sh
# T0 checks (no AWS calls): formatting, validation, and a policy check that nothing opens inbound ports.
check:
	$(TF) fmt -check -recursive .. && $(TF) init -backend=false -input=false >/dev/null && $(TF) validate
	@! grep -REn --include='*.tf' '^[[:space:]]*ingress[[:space:]]*\{' . || (echo "ingress block found: forbidden in this module" && exit 1)
