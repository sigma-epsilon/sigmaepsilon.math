# Installation

The installation process for our library varies depending on whether you are an end user or a developer.

## Installation for End Users

If you are an end user, you can install the library using the following command:

```console
pip install sigmaepsilon.math
```

## Installation for Developers

For developers, the installation process is a little bit more complicated. We use `Poetry` to
manage our dependencies and the project as well. To install the project, you can follow the
steps below:

1. Clone the repository using GitHub Desktop or the command line. In the latter case, we recommend using a secure SSH connection over HTTPS.

2. Install `Poetry` globally:

   ```console
   pip install poetry
   ```

3. Install the library with the necessary optional depencencies by issuing the following command:

   ```console
   poetry install . --with dev,test,doc
   ```

This process will install the library with all dependencies. Note that with `Poetry`, libraries are always installed in editable mode by default. However, If you are working across several solutions in the SigmaEpsilon namespace, and you want them installed in editable mode, the suggested way is to use `Pip`. For instance, if you want to install `sigmaepsilon.core` in editable mode and it is located at the same level as `sigmaepsilon.math`, you issue the following command:

```console
poetry run pip install -e ..\sigmaepsilon.core
```

If the library that you want to install in editable mode is located somewhere else, adjust the path accordingly.