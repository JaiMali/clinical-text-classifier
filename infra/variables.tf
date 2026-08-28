variable "project" {
  description = "Name prefix / tag for all resources"
  type        = string
  default     = "clinical-text-classifier"
}

variable "region" {
  type    = string
  default = "us-east-1"
}

variable "instance_type" {
  description = "t3.micro is free-tier eligible for 12 months on new accounts"
  type        = string
  default     = "t3.micro"
}

variable "ssh_public_key_path" {
  description = "Public key uploaded as the EC2 key pair"
  type        = string
  default     = "~/.ssh/clinical-text-classifier.pub"
}

variable "ssh_ingress_cidr" {
  description = "CIDR allowed to reach port 22 (set to your public IP /32 in terraform.tfvars)"
  type        = string
}
