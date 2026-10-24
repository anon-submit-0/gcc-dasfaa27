#!/usr/bin/env bash
# Download the public Olist e-commerce dataset (MIT license) used by Boundary O.
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d olist-data ]; then
  git clone --depth 1 https://github.com/olist/work-at-olist-data olist-data
fi
ls olist-data/datasets
