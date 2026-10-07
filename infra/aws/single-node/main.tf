# One GPU instance for T2 labs. Access is via SSM Session Manager only: no SSH key, no inbound rules.
# Teardown: `make down` (terraform destroy). Idle auto-stop: see user_data / guardrails.

data "aws_ami" "dlami" {
  count       = var.ami_id == "" ? 1 : 0
  most_recent = true
  owners      = ["amazon"]
  filter {
    # UNVERIFIED name pattern for the DLAMI "Base OSS Nvidia Driver GPU" Ubuntu 22.04 image. If the lookup fails,
    # find the current name in the DLAMI release notes (docs.aws.amazon.com/dlami/) and update it, or set var.ami_id.
    name   = "name"
    values = ["Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*"]
  }
  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

locals {
  ami = var.ami_id != "" ? var.ami_id : data.aws_ami.dlami[0].id
  name = "s2s-${var.lab}-${var.owner}"
}

data "aws_vpc" "default" {
  default = true
}

# No ingress at all. Egress is needed for package installs, the Hugging Face Hub, ECR and the SSM endpoints.
resource "aws_security_group" "node" {
  name_prefix = "${local.name}-"
  description = "S2S GPU node: no inbound; SSM provides shell access"
  vpc_id      = data.aws_vpc.default.id
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "gpu" {
  ami                    = local.ami
  instance_type          = var.instance_type
  iam_instance_profile   = aws_iam_instance_profile.node.name
  vpc_security_group_ids = [aws_security_group.node.id]

  # IMDSv2 only: blocks SSRF-style credential theft from anything you run on the box (e.g. an inference server).
  metadata_options {
    http_tokens                 = "required"
    http_put_response_hop_limit = 2 # 2 so containers on the host can still reach IMDS through the docker bridge
  }

  root_block_device {
    volume_size = var.root_volume_gb
    volume_type = "gp3"
    encrypted   = true
  }

  dynamic "instance_market_options" {
    for_each = var.use_spot ? [1] : []
    content {
      market_type = "spot"
      spot_options {
        instance_interruption_behavior = "stop"
        spot_instance_type             = "persistent"
      }
    }
  }

  # Stopping from inside the instance (the idle watchdog) must stop, not terminate: you keep the disk and the model.
  instance_initiated_shutdown_behavior = "stop"

  user_data = templatefile("${path.module}/user_data.sh.tftpl", { idle_minutes = var.idle_minutes })

  tags = { Name = local.name }
}
