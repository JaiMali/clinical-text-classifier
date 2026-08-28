output "public_ip" {
  description = "Public IPv4 of the instance (changes on stop/start unless an EIP is added)"
  value       = aws_instance.this.public_ip
}

output "public_dns" {
  value = aws_instance.this.public_dns
}

output "url" {
  description = "The running service"
  value       = "http://${aws_instance.this.public_ip}"
}

output "ssh" {
  value = "ssh -i ~/.ssh/clinical-text-classifier ec2-user@${aws_instance.this.public_ip}"
}
