#!/bin/bash
echo "DONT FORGET TO CHECK DEBUG FLAG"

docker-compose up -d classifier-gpu
docker-compose wait classifier-gpu

for i in {10..1}; do
  echo "Shutting down in $i seconds... Ctrl+C to cancel."
  sleep 1
done

echo "Shutting down..."

sudo shutdown -h now