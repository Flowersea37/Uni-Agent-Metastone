#!/bin/bash

HEAD_IP=174.1.59.5
CONTAINER_NAME=verl_test

WORKERS=(
    "174.1.59.4"
)

for IP in "${WORKERS[@]}"; do
    echo "Starting worker: $IP"

    ssh root@$IP "
        docker start ${CONTAINER_NAME} >/dev/null

        docker exec ${CONTAINER_NAME} bash -lc '
            cd /data/xgq/projects/RLs/uni-agent
            unset NCCL_IGNORE_NET_MISMATCH

            ray stop --force || true

            fuser /dev/nvidia* 2>/dev/null \
                | tr \" \" \"\n\" \
                | grep -E \"^[0-9]+$\" \
                | xargs -r kill -9

            ray start \
                --address=${HEAD_IP}:6379 \
                --node-ip-address=${IP} \
                --num-gpus=8
        '
    " &
done

wait

echo "Workers started."