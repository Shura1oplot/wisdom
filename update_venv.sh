#!/bin/bash

set -e

. ./.env

systemctl stop "$SYSTEMD_SERVICE"

test -e .venv.bak && rm -rf .venv.bak
mv .venv .venv.bak
python -m venv .venv

. ./.venv/bin/activate

pip install --no-cache-dir \
    -r requirements.txt \
    -e submodules/LightRAG

systemctl start "$SYSTEMD_SERVICE"
