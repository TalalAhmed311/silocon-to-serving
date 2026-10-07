output "record" { value = var.record_name }
output "health_checks" {
  value = { primary = aws_route53_health_check.primary.id, secondary = aws_route53_health_check.secondary.id }
}
output "alarms" {
  value = [aws_cloudwatch_metric_alarm.primary_down.alarm_name, aws_cloudwatch_metric_alarm.secondary_down.alarm_name]
}
