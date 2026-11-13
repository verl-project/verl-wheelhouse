#!/usr/bin/env bash
set -euo pipefail

# Hosted runners may use zram; leave existing swap devices untouched.
df -h /
free -h
swapon --show
swap_file="$(sudo mktemp /swapfile-wheelhouse.XXXXXX)"
sudo fallocate -l 10G "$swap_file"
sudo chmod 600 "$swap_file"
sudo mkswap "$swap_file"
sudo swapon "$swap_file"
swapon --show
free -h
df -h /
