PORT="${1:-8888}"

GCDOCKER_FLAGS="--ulimit memlock=-1:-1 --net=host --ipc=host $([ -d "/dev/infiniband/" ] && echo "--device=/dev/infiniband") --cap-add=IPC_LOCK -e IPUOF_VIPU_API_HOST=${IPUOF_VIPU_API_HOST} -e IPUOF_VIPU_API_PARTITION_ID=${IPUOF_VIPU_API_PARTITION_ID}"

JUPYTER_COMMAND="python -m jupyter lab --port ${PORT} --no-browser --allow-root --ip 0.0.0.0"

docker run --rm -it -v "$(pwd):/work" -w /work ${GCDOCKER_FLAGS} \
    graphcore/pytorch:3.1.0-ubuntu-20.04 \
    bash -c "SETUP_NO_JUPYTER=1 source setup.sh ; pip install -r .tools/requirements.txt ; ${JUPYTER_COMMAND}"
