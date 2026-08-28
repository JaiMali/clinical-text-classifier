#!/bin/bash
# EC2 bootstrap: Docker + swap. The application container is deployed
# separately (by the GitHub Actions "deploy" job, which SSHes in and runs
# `docker pull ghcr.io/<owner>/clinical-text-classifier && docker run ...`).
set -euxo pipefail

# --- 2 GiB swapfile ---------------------------------------------------------
# t3.micro has only 1 GiB RAM; the DistilBERT checkpoint load briefly needs
# more. Swap absorbs the spike; steady-state RSS is ~340 MiB.
if ! swapon --show | grep -q '/swapfile'; then
  dd if=/dev/zero of=/swapfile bs=1M count=2048
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

# --- Docker ---------------------------------------------------------------
dnf install -y docker
systemctl enable --now docker
usermod -aG docker ec2-user
