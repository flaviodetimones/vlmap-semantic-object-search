#!/bin/bash
set -e

echo "Container started."
echo "Working directory: $(pwd)"

exec "$@"
