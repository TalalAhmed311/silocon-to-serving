# Active-passive DNS failover driven by the in-region prober (platform/failover/prober.py):
#   prober (authenticated, inside each region) → CloudWatch metric S2S/Failover RegionUp{Region}
#   → CloudWatch alarm (per region) → Route 53 CLOUDWATCH_METRIC health check → failover record set.
# No health endpoint is exposed to the internet: Route 53 reads the alarm state, not the service.

resource "aws_cloudwatch_metric_alarm" "primary_down" {
  provider            = aws.primary
  alarm_name          = "s2s-region-down-${var.primary_region}"
  namespace           = "S2S/Failover"
  metric_name         = "RegionUp"
  dimensions          = { Region = var.primary_region }
  statistic           = "Minimum"
  period              = 60
  evaluation_periods  = var.alarm_periods
  threshold           = 1
  comparison_operator = "LessThanThreshold"
  treat_missing_data  = "breaching" # a dead prober means we can't see the region: treat as down
}

resource "aws_cloudwatch_metric_alarm" "secondary_down" {
  provider            = aws.secondary
  alarm_name          = "s2s-region-down-${var.secondary_region}"
  namespace           = "S2S/Failover"
  metric_name         = "RegionUp"
  dimensions          = { Region = var.secondary_region }
  statistic           = "Minimum"
  period              = 60
  evaluation_periods  = var.alarm_periods
  threshold           = 1
  comparison_operator = "LessThanThreshold"
  treat_missing_data  = "breaching"
}

resource "aws_route53_health_check" "primary" {
  type                            = "CLOUDWATCH_METRIC"
  cloudwatch_alarm_name           = aws_cloudwatch_metric_alarm.primary_down.alarm_name
  cloudwatch_alarm_region         = var.primary_region
  insufficient_data_health_status = "Unhealthy"
  tags                            = { Name = "s2s-${var.primary_region}" }
}

resource "aws_route53_health_check" "secondary" {
  type                            = "CLOUDWATCH_METRIC"
  cloudwatch_alarm_name           = aws_cloudwatch_metric_alarm.secondary_down.alarm_name
  cloudwatch_alarm_region         = var.secondary_region
  insufficient_data_health_status = "Unhealthy"
  tags                            = { Name = "s2s-${var.secondary_region}" }
}

resource "aws_route53_record" "primary" {
  zone_id         = var.hosted_zone_id
  name            = var.record_name
  type            = "CNAME"
  ttl             = var.ttl
  records         = [var.primary_endpoint]
  set_identifier  = "primary"
  health_check_id = aws_route53_health_check.primary.id
  failover_routing_policy { type = "PRIMARY" }
}

resource "aws_route53_record" "secondary" {
  zone_id         = var.hosted_zone_id
  name            = var.record_name
  type            = "CNAME"
  ttl             = var.ttl
  records         = [var.secondary_endpoint]
  set_identifier  = "secondary"
  health_check_id = aws_route53_health_check.secondary.id
  failover_routing_policy { type = "SECONDARY" }
}
