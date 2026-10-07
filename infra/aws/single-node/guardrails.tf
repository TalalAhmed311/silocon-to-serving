# Cost guardrails that exist whether or not the idle watchdog works:
#   1. an AWS Budget with email alerts at 80% (forecast) and 100% (actual)
#   2. a CloudWatch alarm that STOPS the instance after 3 h of < 2% CPU (catches a dead watchdog)
module "budget" {
  source       = "../guardrails"
  name         = local.name
  limit_usd    = var.monthly_budget_usd
  email        = var.budget_email
}

resource "aws_cloudwatch_metric_alarm" "idle_stop" {
  alarm_name          = "${local.name}-idle-stop"
  namespace           = "AWS/EC2"
  metric_name         = "CPUUtilization"
  dimensions          = { InstanceId = aws_instance.gpu.id }
  statistic           = "Average"
  period              = 900
  evaluation_periods  = 12 # 12 × 15 min = 3 h
  comparison_operator = "LessThanThreshold"
  threshold           = 2
  alarm_description   = "Backstop: stop the GPU instance after 3 h of near-zero CPU."
  alarm_actions       = ["arn:aws:automate:${var.region}:ec2:stop"]
}
