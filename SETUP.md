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

5. Install Flash Attention 3 (this can take some time):
    ```bash
    # Ensure you are on a node with the container loaded and the the virtual environment is activated
    uv pip install setuptools packaging ninja
    cd /iopsstor/scratch/cscs/$USER/
    git clone https://github.com/Dao-AILab/flash-attention.git
    cd flash-attention/hopper
    MAX_JOBS=25 python setup.py install
    # We need to patch the installation because Megatron-LM and transformer engine expect the package at a different location
    cp /iopsstor/scratch/cscs/$USER/.venv-gipfelturm/lib/python-3.12/site-packages/flash_attn_interface.py /iopsstor/scratch/cscs/$USER/.venv-gipfelturm/lib/python-3.12/site-packages/flash_attn_3/flash_attn_interface.py
    touch /iopsstor/scratch/cscs/$USER/.venv-gipfelturm/lib/python-3.12/site-packages/flash_attn_3/__init__.py
    ```

5. Check that everything is installed correctly:
    ```bash
    pip list -v
    ```

## Explanation

The reasons why this is kind of complicated originates from the fact that uv **really** **really** doesn't want to use the system packages. However, for us they are highly optimized compiled libraries. Although we seed the uv venv and could use pip, Megatron-LM was packaged using uv which forces us to use the pip interface. That is why we have a patch to force uv to use the system packages. In addition, as we use the dev dependencies of Megatron-LM (for access to MoE and Quantization), our setup requires a lot of specifically compiled CUDA code. 