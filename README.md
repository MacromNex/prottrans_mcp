# ProtTrans MCP

**Protein fitness modeling using ProtTrans language models via Docker**

An MCP (Model Context Protocol) server for protein fitness prediction with 4 core tools:
- Extract ProtTrans embeddings from protein sequences
- Calculate ProtBERT log-likelihood scores for mutations
- Train regression fitness models with 5-fold cross-validation
- Predict fitness for new protein sequences using trained models

## Quick Start with Docker

### Approach 1: Pull Pre-built Image from GitHub

The fastest way to get started. A pre-built Docker image is automatically published to GitHub Container Registry on every release.

```bash
# Pull the latest image
docker pull ghcr.io/macromnex/prottrans_mcp:latest

# Register with Claude Code (runs as current user to avoid permission issues)
claude mcp add prottrans -- docker run -i --rm --user `id -u`:`id -g` --gpus all --ipc=host -v `pwd`:`pwd` ghcr.io/macromnex/prottrans_mcp:latest
```

**Note:** Run from your project directory. `` `pwd` `` expands to the current working directory.

**Requirements:**
- Docker with GPU support (`nvidia-docker` or Docker with NVIDIA runtime)
- Claude Code installed

That's it! The ProtTrans MCP server is now available in Claude Code.

---

### Approach 2: Build Docker Image Locally

Build the image yourself and install it into Claude Code. Useful for customization or offline environments.

```bash
# Clone the repository
git clone https://github.com/MacromNex/prottrans_mcp.git
cd prottrans_mcp

# Build the Docker image
docker build -t prottrans_mcp:latest .

# Register with Claude Code (runs as current user to avoid permission issues)
claude mcp add prottrans -- docker run -i --rm --user `id -u`:`id -g` --gpus all --ipc=host -v `pwd`:`pwd` prottrans_mcp:latest
```

**Note:** Run from your project directory. `` `pwd` `` expands to the current working directory.

**Requirements:**
- Docker with GPU support
- Claude Code installed
- Git (to clone the repository)

**About the Docker Flags:**
- `-i` — Interactive mode for Claude Code
- `--rm` — Automatically remove container after exit
- `` --user `id -u`:`id -g` `` — Runs the container as your current user, so output files are owned by you (not root)
- `--gpus all` — Grants access to all available GPUs
- `--ipc=host` — Uses host IPC namespace for PyTorch shared memory
- `-v` — Mounts your project directory so the container can access your data

---

## Verify Installation

After adding the MCP server, you can verify it's working:

```bash
# List registered MCP servers
claude mcp list

# You should see 'prottrans' in the output
```

In Claude Code, you can now use all 4 ProtTrans tools:
- `prottrans_extract_embeddings`
- `prottrans_calculate_llh`
- `prottrans_train_fitness_model`
- `prottrans_predict_fitness`

---

## Next Steps

- **Detailed documentation**: See [detail.md](detail.md) for comprehensive guides on:
  - Available MCP tools and parameters
  - Local Python environment setup (alternative to Docker)
  - Example workflows and use cases
  - Data format requirements
  - Troubleshooting

---

## Usage Examples

Once registered, you can use the ProtTrans tools directly in Claude Code. Here are some common workflows:

### Example 1: Train a Fitness Model

```
Can you help train a ProtTrans model for data at /path/to/example/ and save it to /path/to/results/prot-t5_fitness using the prottrans MCP server with ProtT5-XL model. Please create the embeddings first if not ready.
```

### Example 2: Calculate Log-Likelihoods

```
Can you help calculate ProtBERT likelihood for data at /path/to/data.csv with wild-type sequence at /path/to/wt.fasta using the prottrans MCP server?
```

### Example 3: Full Fitness Modeling Workflow

```
I have protein variant data at /path/to/variants.csv with log_fitness column. Please:
1. Extract ProtT5-XL embeddings using prottrans_extract_embeddings
2. Train an SVR fitness model using prottrans_train_fitness_model with 5-fold CV
3. Report the mean Spearman correlation performance
```

---

## Troubleshooting

**Docker not found?**
```bash
docker --version  # Install Docker if missing
```

**GPU not accessible?**
- Ensure NVIDIA Docker runtime is installed
- Check with: `docker run --gpus all ubuntu nvidia-smi`

**Claude Code not found?**
```bash
# Install Claude Code
npm install -g @anthropic-ai/claude-code
```

**Out of GPU memory?**
- ProtT5-XL requires 8-16 GB VRAM
- Use `device: "cpu"` for CPU inference (slower)
- Use ProtAlbert for lower memory requirements

---

## License

MIT — Based on [ProtTrans](https://github.com/agemagician/ProtTrans) by Elnaggar et al.
