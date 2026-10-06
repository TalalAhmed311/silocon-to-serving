output "instance_id" {
  value = aws_instance.gpu.id
}

output "ssm_shell" {
  value = "aws ssm start-session --region ${var.region} --target ${aws_instance.gpu.id}"
}

output "port_forward_8000" {
  description = "Reach a server bound to 127.0.0.1:8000 on the instance from your laptop's localhost:8000."
  value       = "aws ssm start-session --region ${var.region} --target ${aws_instance.gpu.id} --document-name AWS-StartPortForwardingSession --parameters portNumber=8000,localPortNumber=8000"
}

output "ami" {
  value = local.ami
}
