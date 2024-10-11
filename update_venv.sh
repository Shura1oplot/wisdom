#!/bin/bash

set -e

. ./.env

systemctl stop "$SYSTEMD_SERVICE"
rm -rf .venv
python -m venv .venv
. ./.venv/bin/activate
pip install -r requirements.txt
systemctl start "$SYSTEMD_SERVICE"

