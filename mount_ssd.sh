#!/bin/bash

# Exit on any error
set -e

echo "Setting up instance store SSD for Docker (ephemeral storage)..."

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root (use sudo)"
    exit 1
fi

# Find instance store NVMe device
SSD_DEVICE=$(lsblk | grep nvme | grep -v nvme0n1 | awk '{print "/dev/"$1}' | head -1)

if [ -z "$SSD_DEVICE" ]; then
    echo "No instance store SSD found. Exiting."
    exit 1
fi

echo "Found SSD device: $SSD_DEVICE"

# Stop Docker
echo "Stopping Docker..."
systemctl stop docker

# Create mount point
MOUNT_POINT="/mnt/ssd"
mkdir -p $MOUNT_POINT

# Format and mount the SSD
echo "Formatting and mounting SSD..."
mkfs -t ext4 -F $SSD_DEVICE
mount $SSD_DEVICE $MOUNT_POINT

# Create Docker directory
DOCKER_DIR="$MOUNT_POINT/docker"
mkdir -p $DOCKER_DIR

# Configure Docker to use the new location
echo "Configuring Docker..."
mkdir -p /etc/docker

cat > /etc/docker/daemon.json <<EOF
{
    "data-root": "$DOCKER_DIR",
    "storage-driver": "overlay2"
}
EOF

# Start Docker
echo "Starting Docker..."
systemctl daemon-reload
systemctl start docker

# Verify
echo "Docker is now using: $(docker info | grep 'Docker Root Dir')"

echo ""
echo "Setup complete! Docker is now using the instance store SSD."
echo "Remember: All data will be lost when the instance stops."