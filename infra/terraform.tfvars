# CIDR allowed to reach port 22. Currently open to all (key-only auth; no
# password login). Tighten to specific IPs and re-apply if you want:
#   ssh_ingress_cidr = "203.0.113.4/32"   # then: terraform apply
# Find your current IP with: curl -s https://checkip.amazonaws.com
ssh_ingress_cidr = "0.0.0.0/0"
