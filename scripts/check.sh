#!/usr/bin/env bash

set -eo pipefail

uv run ruff format --check .
uv run ruff check .
uv run pytest
