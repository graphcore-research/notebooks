# Development

You can test out your notebooks in a few ways:

 1. Standalone Jupyter server, `.dev/run_server.sh` (uses Docker)
 2. VS Code devcontainers "Open in Container" (uses Docker)
 3. Using a virtual environment (no Docker)

**1. Standalone**

The script `.dev/run_server.sh` will start a server that looks quite like Paperspace Gradient. You may need to forward ports using `ssh -L localhost:LOCAL_PORT:localhost:REMOTE_PORT` to access it. This automatically forwards a `/datasets` directory into the image, and should work with our without IPUs.

**2. Devcontainers**

If using devcontainers, use "Open in Container" to build the container & open the project. We haven't configured this for using real IPUs, as it is hard to support both cases in `devcontainer.json`. (Note that for git commands, you might want to use "Create new integrated terminal (local)".)

We'd suggest also using a virtual env, so that it's easy to run `setup.sh` to load the same environment variables as Paperspace Gradient. Make sure the Jupyter kernel is set to use the Python interpreter from the venv.

```bash
python -m venv --system-site-packages .venv
echo 'SETUP_NO_JUPYTER=1 source $VIRTUAL_ENV/../setup.sh' >> .venv/bin/activate
# unless/until you get devcontainers working with real IPUs:
echo 'export POPTORCH_IPU_MODEL=1' >> .venv/bin/activate
```

**3. VirtualEnv**

Since this doesn't run in the same Docker image as you will use on Paperspace, it may have various differences, but could also be less friction to use. If using VS Code, make sure the Jupyter kernel is set to use the Python interpreter from the venv.

```bash
python3 -m venv .venv

# In .venv/bin/activate
    POPLAR_SDK_PATH=/opt/gc/poplar_sdk-ubuntu_20_04-3.0.0+1145-1b114aac3a
    source "${POPLAR_SDK_PATH}/enable"
    source $POPLAR_SDK_PATH/popart*/enable.sh
    export PATH="${PATH}:${POPLAR_SDK_ENABLED}/bin"
    SETUP_NO_JUPYTER=1 source $VIRTUAL_ENV/../setup.sh

source .venv/bin/activate
pip install $POPLAR_SDK_ENABLED/../poptorch*.whl

# Optionally set $PUBLIC_DATASET_DIR to point to your datasets
```

# Contributing

These instructions are intended for members of the Graphcore Research team.

To release a new notebook:

1. Obtain "ready to release" approval from a Research lead.
2. Push to the internal repo and request a review (see checklist below).
3. Push to public and verify on Paperspace.

**Review**

 - (Author) Check that there are no references to unreleased products or customers (including in any git history).
 - (Author) Did you write all the code, or have you clearly attributed the author?
 - Perform an IP review according to the current process.
 - New notebooks added to README.md?
 - Once you're happy, "approve" and let the submitter merge the PR.
