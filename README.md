# Research Notebooks

| Link | Description |
| --- | --- |
| [![Run on Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/graphcore-research/research-notebooks?container=graphcore%2Fpytorch-jupyter%3A3.1.0-ubuntu-20.04&machine=Free-IPU-POD4&file=%2Fhello_pytorch%2FHelloPyTorch.ipynb) | PyTorch example |

## Development

See [Contributing.md](Contributing.md).

You can test out your notebooks in a few ways:

 1. Standalone Jupyter server, `.tools/run_server.sh` on Graphcore CL1 (uses Docker)
 2. VS Code devcontainers "Open in Container" (uses Docker)
 3. Using a virtual environment (no Docker)

**1. Standalone**

The script `.tools/run_server.sh` will start a server that looks quite like Paperspace Gradient. You may need to forward ports using `ssh -L localhost:LOCAL_PORT:localhost:REMOTE_PORT` to access it. This automatically forwards a `/datasets` directory into the image, and should work with our without IPUs.

**2. Devcontainers**

We'd suggest also using a virtual env, so that it's easy to run `setup.sh` to load the same environment variables as Paperspace Gradient. We haven't configured this for using real IPUs, as it is hard to support both cases in `devcontainer.json`.

```bash
python -m venv --system-site-packages .venv
echo 'SETUP_NO_JUPYTER=1 source $VIRTUAL_ENV/../setup.sh' >> .venv/bin/activate
# unless/until you get devcontainers working with real IPUs:
echo 'export POPTORCH_IPU_MODEL=1' >> .venv/bin/activate
```

**3. VirtualEnv**

Since this doesn't run in the same Docker image as you will use on Paperspace, it may have various differences, but could also be less friction to use.

```bash
python3 -m venv .venv

# In .venv/bin/activate
    POPLAR_SDK_PATH=/opt/gc/poplar_sdk-ubuntu_20_04-3.0.0+1145-1b114aac3a
    source "${POPLAR_SDK_PATH}/enable"
    source $POPLAR_SDK_PATH/popart*/enable.sh
    export PATH="${PATH}:${POPLAR_SDK_ENABLED}/bin"
    SETUP_NO_JUPYTER=1 source $VIRTUAL_ENV/../setup.sh

source .venv/bin/activate
pip install $POPLAR_SDK_ENABLED/poptorch*.whl

# Optionally set $PUBLIC_DATASET_DIR to point to your datasets
```

## License

Copyright (c) 2023 Graphcore Ltd. Licensed under the MIT License.

The included code is released under an MIT license, (see [LICENSE](LICENSE)).
