#!/bin/bash
# Launch the TFG Docker environment with X11 forwarding enabled.
# Usage:  ./docker/start.sh          (starts container)
#         ./docker/start.sh build    (rebuilds and starts)

set -e
cd "$(dirname "$0")"

# Allow Docker containers to access the host X11 display
xhost +local:docker 2>/dev/null || true

if [ "$1" = "build" ]; then
    docker compose up --build -d
else
    docker compose up -d
fi

echo ""
echo "Container started. Attach with:"
echo "  docker exec -it tfg-sim bash"
