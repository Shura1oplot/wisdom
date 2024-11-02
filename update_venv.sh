#!/bin/bash

set -e

. ./.env

if [ -e .venv.bak ]; then
    rm -rf .venv.bak
fi

systemctl stop "$SYSTEMD_SERVICE"
mv .venv .venv.bak
python -m venv .venv
. ./.venv/bin/activate
pip install --no-cache-dir -r requirements.txt
systemctl start "$SYSTEMD_SERVICE"
