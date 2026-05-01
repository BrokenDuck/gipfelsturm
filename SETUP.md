# Setup commands to get everything up and running

1. Apply the git patches:
    ```bash
    cd $MEGATRON_LM_DIR && git checkout -- . && git apply $WORKDIR/patches/*.patch        
    ```

2. First, start a session in the container:
    ```bash
    srun -p debug -t 00:30:00 -A lsaie-ss26 --mpi=pmix --network=disable_rdzv_get --environment=alps3 --pty bash
    ```

3. Follow https://docs.cscs.ch/build-install/python/#installing-venv-on-top-of-a-uenv-view. As the venv will contain a lot of files, use iopstor:
    ```bash
    export PYTHONUSERBASE="$(dirname "$(dirname "$(which python)")")"
    uv venv --python $(which python) --system-site-packages --seed --relocatable --link-mode=copy /iopsstor/scratch/cscs/$USER/.venv-gipfelturm
    ```

4. Install Megatron-LM (this can take some time):
    ```bash
    source /iopsstor/scratch/cscs/$USER/.venv-gipfelturm/bin/activate
    uv pip install setuptools pybind11 packaging
    cd Megatron-LM && uv pip install --compile-bytecode --no-build-isolation --link-mode=copy --editable ".[dev,training]"
    ```

## Explanation

The reasons why this is kind of complicated originates from the fact that uv **really** **really** doesn't want to use the system packages. However, for us they are highly optimized compiled libraries. Although we seed the uv venv and could use pip, Megatron-LM was packaged using uv which forces us to use the pip interface. That is why we have a patch to force uv to use the system packages. In addition, as we use the dev dependencies of Megatron-LM (for access to MoE and Quantization), our setup requires a lot of specifically compiled CUDA code. 