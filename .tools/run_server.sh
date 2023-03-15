PORT="${1:-8888}"

GCDOCKER_FLAGS="--ulimit memlock=-1:-1 --net=host --ipc=host $([ -d "/dev/infiniband" ] && echo "--device=/dev/infiniband") --cap-add=IPC_LOCK -e IPUOF_VIPU_API_HOST=${IPUOF_VIPU_API_HOST} -e IPUOF_VIPU_API_PARTITION_ID=${IPUOF_VIPU_API_PARTITION_ID}"
MOUNT_FLAGS="$([ -d "/scratch_ai-datasets/paperspace" ] && echo "-v /scratch_ai-datasets/paperspace:/datasets:ro")"

docker run --rm -it -v "$(pwd):/work" -w /work ${GCDOCKER_FLAGS} ${MOUNT_FLAGS} \
    graphcore/pytorch:3.1.0-ubuntu-20.04 \
    .tools/.server.sh "${PORT}"
