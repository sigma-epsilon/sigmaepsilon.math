---
myst:
  html_meta:
    description: "How to set up a development environment for sigmaepsilon.math with uv, with or without Numba, and how to install sibling SigmaEpsilon libraries in editable mode."
---

# Installation for Developers

For developers, the installation process is a little bit more complicated. We use `uv` to
manage our dependencies and the project as well. To install the project, you can follow the
steps below:

1. Clone the repository using GitHub Desktop or the command line. In the latter case, we recommend using a secure SSH connection over HTTPS.

2. Install `uv` globally (see the [uv documentation](https://docs.astral.sh/uv/getting-started/installation/) for other options):

   ```shell
   pip install uv
   ```

3. Install the library with Numba and the necessary optional dependency groups by issuing the following command:

   ```shell
   uv sync --extra numba --group dev --group test --group docs
   ```

This process will create a virtual environment in `.venv` and install the library with all dependencies.
Numba is an optional dependency of the library. To run the tests without it, the way the
`run_tests_no_numba` CI job does, leave out `--extra numba` (a subsequent `uv sync` without the
extra uninstalls Numba from the environment).
Note that with `uv`, the project itself is always installed in editable mode.
If you are working across several solutions in the SigmaEpsilon namespace, and you want them installed in editable mode,
you can install them into the same environment with `uv pip`. For instance, if you want to install `sigmaepsilon.core` in editable mode and it is located at the same level as
`sigmaepsilon.math`, you issue the following command:

```shell
uv pip install -e ../sigmaepsilon.core
```

Keep in mind that a subsequent `uv sync` restores the locked versions, so you need to repeat this step after syncing.

If the library that you want to install in editable mode is located somewhere else, adjust the path accordingly.
