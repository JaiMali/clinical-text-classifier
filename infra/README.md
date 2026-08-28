# infra/ — EC2 deployment (Terraform)

Provisions a single free-tier EC2 instance to run the classifier container.

**What it creates** (all free-tier eligible for 12 months on a new account):

| Resource | Detail |
|---|---|
| `aws_instance` | `t3.micro`, Amazon Linux 2023, 20 GiB gp3 root |
| `aws_security_group` | inbound `22` from your IP only, `80` from anywhere |
| `aws_key_pair` | uploads `~/.ssh/clinical-text-classifier.pub` |

`user_data.sh` installs Docker and a 2 GiB swapfile on first boot. The
application container itself is deployed by the GitHub Actions `deploy` job.

## Usage

```bash
cd infra
terraform init
terraform plan       # review
terraform apply      # ~2 min; prints public_ip / url / ssh

# tear everything down (stops all cost):
terraform destroy
```

## Notes

- **Cost:** $0 within the 12-month free tier. `terraform destroy` removes
  everything; stopping the instance (`aws ec2 stop-instances`) pauses compute
  cost but the EBS volume still counts against the 30 GiB free allowance.
- **Public IP changes** on stop/start. Add an Elastic IP if you need a stable
  URL (free while attached to a *running* instance).
- **`terraform.tfvars`** holds your SSH ingress IP — update it if your ISP
  reassigns your address.
