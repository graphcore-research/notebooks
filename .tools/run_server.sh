PORT="${1:-8888}"

docker run --rm -it \
    -v "$(pwd):/work" -w /work -p "${PORT}:${PORT}" \
    graphcore/pytorch:3.1.0-ubuntu-20.04 \
    bash -c "SETUP_NO_JUPYTER=1 source setup.sh ; pip install ipywidgets jupyterlab ; python -m jupyter lab --port ${PORT} --no-browser --allow-root --ip 0.0.0.0"
