#!/usr/bin/env bash
# Privacy audit: must print nothing.
cd "$(dirname "${BASH_SOURCE[0]}")/.." 
grep -rInE '/home/|/mnt/|ghp_|hf_[A-Za-z0-9]{10}|sk-[A-Za-z0-9]{10}|[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}|tailscale|\.ts\.net|@[a-z0-9-]+\.(com|org|net)' \
  --exclude-dir=.git --exclude=audit.sh . | grep -v -E 'noreply\.github\.com|127\.0\.0\.1|0\.0\.0\.0' \
  | grep -v -E '^\./docker/requirements-[a-z0-9]+\.lock:[0-9]+:[A-Za-z0-9_.-]+==[0-9A-Za-z.+]+$' 
exit 0
