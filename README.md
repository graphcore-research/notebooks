# Research Notebooks

| Link | Description |
| --- | --- |
| - | PyTorch example |

## Development

See [Contributing.md](Contributing.md).

We recommend testing with either:

 - VS Code devcontainers "Open in Container"
 - `.tools/run_server.sh` on Graphcore CL1

If using VS Code (with devcontainers), we'd suggest also using a virtual env, so that it's easy to run `setup.sh` to load the same environment variables as Paperspace Gradient.

```bash
python -m venv --system-site-packages .venv
echo 'SETUP_NO_JUPYTER=1 source $VIRTUAL_ENV/../setup.sh' >> .venv/bin/activate
# if not using real IPUs, add:
echo 'export POPTORCH_IPU_MODEL=1' >> .venv/bin/activate
```

## License

Copyright (c) 2023 Graphcore Ltd. Licensed under the MIT License.

The included code is released under an MIT license, (see [LICENSE](LICENSE)).
