TF ?= terraform
VARS ?= terraform.tfvars
PRIMARY_CTX ?= s2s-primary
SECONDARY_CTX ?= s2s-secondary
.PHONY: init up down check status drill-kill drill-restore
init:
	$(TF) init
up: init
	$(TF) apply -var-file=$(VARS)
down:
	$(TF) destroy -var-file=$(VARS)
check:
	$(TF) fmt -check && $(TF) init -backend=false -input=false >/dev/null && $(TF) validate
status:
	@for id in $$($(TF) output -json health_checks | python3 -c 'import json,sys;print(" ".join(json.load(sys.stdin).values()))'); do \
	  aws route53 get-health-check-status --health-check-id $$id --query 'HealthCheckObservations[0].StatusReport.Status' --output text; done
# The game day "kill": scale the primary gateway to zero (the GPU pods keep running, so restore is fast).
drill-kill:
	@date +%s > .kill_t && kubectl --context $(PRIMARY_CTX) -n s2s scale deploy/gateway --replicas=0 && echo "killed at $$(cat .kill_t)"
drill-restore:
	kubectl --context $(PRIMARY_CTX) -n s2s scale deploy/gateway --replicas=2
