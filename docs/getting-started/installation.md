# Installation

tform uses [uv](https://docs.astral.sh/uv/) as its package and environment manager. uv is a fast, modern Python tool that replaces pip, venv, and pyenv in a single command.

---

## Step 1 — Install uv

Choose the command for your operating system.

=== "macOS / Linux"

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

    Then restart your terminal (or run `source ~/.bashrc` / `source ~/.zshrc`).

=== "Windows (PowerShell)"

    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

    Then restart your terminal.

=== "Homebrew (macOS)"

    ```bash
    brew install uv
    ```

=== "pip (any OS)"

    ```bash
    pip install uv
    ```

Verify installation:

```bash
uv --version
```

You should see something like `uv 0.5.x`. If the command is not found, check that `~/.cargo/bin` (the default install path) is on your `PATH`.

Full documentation: [docs.astral.sh/uv/getting-started/installation](https://docs.astral.sh/uv/getting-started/installation/)

---

## Step 2 — Clone the repository

```bash
git clone https://github.com/BioMedAI-UCSC/terraforming
cd terraforming
```

---

## Step 3 — Install dependencies

```bash
uv sync
```

This creates a `.venv/` directory in the project root and installs all dependencies for both the `terraforming` physics package and the `tform` CLI. You do not need to create a virtual environment manually.

---

## Step 4 — Verify tform works

```bash
uv run tform --version
uv run tform man mars
```

Using `uv run` is the recommended way to call tform — it automatically uses the project's virtual environment without you needing to activate it.

---

## Activating the virtual environment (optional)

If you prefer to activate the environment so you can run `tform` directly without `uv run`:

=== "macOS / Linux"

    ```bash
    source .venv/bin/activate
    tform --version
    ```

=== "Windows (PowerShell)"

    ```powershell
    .venv\Scripts\Activate.ps1
    tform --version
    ```

=== "Windows (CMD)"

    ```cmd
    .venv\Scripts\activate.bat
    tform --version
    ```

To deactivate:

```bash
deactivate
```

---

## Troubleshooting

### `tform: command not found`

The most common cause is that the virtual environment is not active and you are not using `uv run`. Fix with either:

```bash
# Option A: always prefix with uv run
uv run tform man mars

# Option B: activate the venv first, then use tform directly
source .venv/bin/activate
tform man mars
```

### `uv: command not found` after installation

The uv installer adds itself to `~/.local/bin` (Linux) or `~/.cargo/bin` (macOS). Make sure that directory is on your `PATH`:

```bash
# Add to ~/.bashrc or ~/.zshrc
export PATH="$HOME/.local/bin:$PATH"

# Then reload
source ~/.bashrc
```

On Windows, restart your terminal after installation — the installer updates the system PATH automatically.

### `uv sync` fails with Python version error

tform requires Python 3.12 or later. uv can install the right Python version automatically:

```bash
uv python install 3.12
uv sync
```

### `.venv` exists but tform still not found

The venv may be stale. Remove it and resync:

```bash
rm -rf .venv
uv sync
```

### `ModuleNotFoundError: No module named 'torch'`

PyTorch was not installed correctly. Run:

```bash
uv sync --reinstall
```

If you need a CUDA-enabled build of PyTorch, install it manually after sync:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## GPU support (optional)

The simulator uses PyTorch for all tensor operations. If a CUDA GPU is available, pass `device="cuda"` when constructing objects directly in Python:

```python
from src.celestials import Mars
from src.engine import TimeController, Accuracy

planet = Mars(device="cuda")
tc = TimeController(planet, accuracy=Accuracy.ACCURATE)
```

The CLI always runs on CPU; GPU acceleration is only available through the Python API.
