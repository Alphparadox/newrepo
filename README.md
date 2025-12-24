# LLaVA Visual Reasoning Benchmark on HPC Cluster

## 📋 Table of Contents
- [Overview](#overview)
- [Beginner Quick Start](#beginner-quick-start)
- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Run on HPC](#run-on-hpc)
- [Check Results](#check-results)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)

> 📖 **For detailed code explanation, see [CODE_EXPLANATION.md](CODE_EXPLANATION.md)**

---

## 🎯 Overview

This project evaluates the **LLaVA-1.5-13B** (Large Language and Vision Assistant) model on visual reasoning tasks. It tests the model's ability to understand and apply visual transformations like rotation, resizing, color changes, counting, and reflection.

### What is LLaVA?
LLaVA is a multimodal AI model that can understand both images and text. It combines a vision encoder (to process images) with a large language model (to generate text responses).

### What is an HPC Cluster?
A High-Performance Computing (HPC) cluster is a group of powerful computers (nodes) connected together. It's used for running computationally intensive tasks like training and testing large AI models. Your cluster has GPUs (H100) which accelerate AI model processing.

---

## 🚀 Beginner Quick Start

Follow these steps exactly. Copy and paste the commands.

1) Connect to your HPC:
```bash
ssh your_username@your_cluster_address
```

2) Create environment and install packages:
```bash
conda create -n kiva_env python=3.10 -y
conda activate kiva_env
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install transformers pillow pandas matplotlib numpy accelerate
```

3) Get the project:
```bash
cd ~
git clone https://github.com/Alphparadox/newrepo.git
cd newrepo
mkdir -p logs
```

4) Submit the job:
```bash
sbatch anil.sbatch
```

5) Watch the logs:
```bash
tail -f logs/llava_test_*.out
```

6) See results:
```bash
cat output/multi_kiva/all_results.csv
```

---

## 📦 Prerequisites

### Required Software:
1. **Access to HPC Cluster** with:
   - SLURM job scheduler
   - GPU nodes (H100 or similar)
   - Conda/Miniconda installed

2. **Python Packages:**
   - `torch` (PyTorch - deep learning framework)
   - `transformers` (Hugging Face library for AI models)
   - `pillow` (image processing)
   - `pandas` (data analysis)
   - `matplotlib` (plotting)
   - `numpy` (numerical computing)

### Storage Requirements:
- **Disk Space:** ~30 GB (for model + images)
- **Memory (RAM):** 64 GB minimum
- **GPU Memory:** 40 GB+ (for 13B model)

---

## 📁 Project Structure

```
newrepo/
│
├── sample.py               # Main evaluation script
├── helper.py               # Utility functions for data processing
├── anil.sbatch             # SLURM job submission script
│
├── multi_image/            # Multi-image presentation format
│   └── test/               # Test data JSON files
│       ├── 2DRotation.json
│       ├── Colour.json
│       ├── Counting.json
│       ├── Reflect.json
│       └── Resize.json
│
└── output/                 # Results directory
    └── multi_kiva/
        └── all_results.csv # Evaluation results
```

---

## 🧰 Prerequisites

This project needs:

- SLURM on the cluster
- 1× GPU (H100) and ~64 GB RAM
- Conda/Miniconda
- Python packages: torch, transformers, pillow, pandas, matplotlib, numpy

Tip: Check GPU and CUDA with:
```bash
nvidia-smi
```

---

## 🏃 Run on HPC

### Method 1: Submit SLURM Job (Recommended)

This runs your code in the background on the cluster:

```bash
cd ~/newrepo
sbatch anil.sbatch
```

### Method 2: Interactive Session (For Testing)

This gives you a GPU node interactively for debugging:

```bash
sinteractive --partition=gpu --gres=shard:H100:1 --mem=64G --cpus-per-task=4
conda activate kiva_env
python sample.py
```

---

## 👀 Monitor Your Job

### Check Job Status

```bash
# View all your jobs
squeue -u your_username

# View detailed job info
scontrol show job 12345  # Replace with your job ID
```

**Job States:**
- `PD` (Pending): Waiting for resources
- `R` (Running): Currently executing
- `CG` (Completing): Finishing up
- `CD` (Completed): Finished successfully
- `F` (Failed): Encountered an error

### View Live Output

```bash
# Watch the output log in real-time
tail -f logs/llava_test_12345.out

# Press Ctrl+C to stop watching
```

### Cancel a Job

```bash
scancel 12345  # Replace with your job ID
```

### Check GPU Usage

If you have an interactive session or the job is running:

```bash
nvidia-smi
```

This shows:
- GPU utilization %
- Memory usage
- Running processes

---

## 📦 Check Results

### Results CSV File

Location: `output/multi_kiva/all_results.csv`

**Columns:**
```csv
img_id,transform,variation,cross_domain_answer,cross_domain_ground_truth,within_domain_answer,within_domain_ground_truth,extrapolation_answer,extrapolation_ground_truth
```

**Example Row:**
```csv
2DRotation+90_0_0,2DRotation,0,"(1) Rotate object...","(1)","(1) Rotate 90 degrees...","(1)","(B)","(B)"
```

### Read the CSV

1. **`img_id`**: Unique identifier for each test case
   - Format: `Transform_InputValue_TrialNumber`
   - Example: `2DRotation+90_0_0` = 90° rotation, input value 0, trial 0

2. **`*_answer`**: Model's prediction
3. **`*_ground_truth`**: Correct answer
4. **Match = Correct, Mismatch = Incorrect**

### Accuracy Summary

The script calculates:
- **Per-concept accuracy** (e.g., 2DRotation: 85%)
- **Per-task accuracy** (e.g., Cross-Domain: 90%)
- **Overall accuracy** across all tests

---

## 🐛 Troubleshooting

### Problem 1: Job Stays in Pending State

**Possible Causes:**
- Cluster is busy (no available GPUs)
- Resource request too large
- Wrong partition name

**Solutions:**
```bash
# Check cluster status
sinfo

# Check your position in queue
squeue -u your_username

# Reduce resources if needed (edit anil.sbatch)
#SBATCH --mem=32G  # Instead of 64G
```

### Problem 2: Out of Memory Error

**Error Message:** `CUDA out of memory`

**Solutions:**

1. Use model quantization (edit `sample.py`):
```python
from transformers import BitsAndBytesConfig

model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    device_map="auto"
)
```

2. Request more GPU memory in SLURM:
```bash
#SBATCH --mem=128G
```

### Problem 3: Module Not Found Error

**Error Message:** `ModuleNotFoundError: No module named 'transformers'`

**Solution:**
```bash
# Make sure conda environment is activated
conda activate kiva_env

# Reinstall the package
pip install transformers
```

### Problem 4: wget or unzip Command Not Found

**Solution:**
```bash
# Check if wget is available
which wget

# If not, ask your cluster admin to install it
# Or modify sample.py to use curl:
os.system(f"curl -o '{prefix}{presentation_type}_image.zip' 'https://storage.googleapis.com/...'")
```

### Problem 5: Permission Denied

**Error Message:** `Permission denied: 'output/multi_kiva'`

**Solution:**
```bash
# Create the directory manually
mkdir -p output/multi_kiva

# Check permissions
ls -la output/
```

---

## ❓ FAQ

| Error | Cause | Solution |
|-------|-------|----------|
| `sbatch: command not found` | SLURM not installed | Contact cluster admin or use `python sample.py` directly |
| `conda: command not found` | Conda not in PATH | Add to PATH: `export PATH=~/miniconda3/bin:$PATH` |
| `ImportError: cannot import name 'AutoProcessor'` | Old transformers version | `pip install --upgrade transformers` |
| `RuntimeError: No GPU available` | GPU not allocated | Check SLURM script GPU request line |
| `Connection timeout` downloading model | Network issues | Download model separately (see below) |

### Pre-download the model

If network is slow, download the model before running:

```bash
python -c "from transformers import AutoProcessor, LlavaForConditionalGeneration; \
           AutoProcessor.from_pretrained('llava-hf/llava-1.5-13b-hf'); \
           LlavaForConditionalGeneration.from_pretrained('llava-hf/llava-1.5-13b-hf')"
```

---

## 📚 Additional Resources

### Learn More About:

- **SLURM Commands**: https://slurm.schedmd.com/quickstart.html
- **Conda Basics**: https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html
- **PyTorch**: https://pytorch.org/tutorials/
- **Transformers Library**: https://huggingface.co/docs/transformers/
- **LLaVA Model**: https://huggingface.co/llava-hf/llava-1.5-13b-hf

### Getting Help

1. **Check logs first**: `cat logs/llava_test_*.err`
2. **Read error messages carefully** - they usually indicate the problem
3. **Ask your cluster admin** about cluster-specific settings
4. **Search error messages online** with "HPC SLURM" added to query

---

## 🎓 Understanding HPC Concepts

### What is SLURM?

**SLURM** (Simple Linux Utility for Resource Management) is a job scheduler. Think of it as a "ticket system" for the cluster:

1. You submit a job (your code) → Get a ticket (job ID)
2. SLURM waits for resources to be available
3. When ready, it runs your job on an available node
4. You get results when it's done

### Key SLURM Commands

| Command | Purpose | Example |
|---------|---------|---------|
| `sbatch` | Submit a job | `sbatch anil.sbatch` |
| `squeue` | Check job status | `squeue -u username` |
| `scancel` | Cancel a job | `scancel 12345` |
| `sinfo` | View cluster info | `sinfo` |
| `scontrol` | Detailed job info | `scontrol show job 12345` |

### Understanding Resource Requests

```bash
#SBATCH --cpus-per-task=4   # 4 CPU cores
#SBATCH --mem=64G           # 64 GB RAM
#SBATCH --gres=shard:H100:1 # 1 GPU (H100 model)
```

**Balance:** More resources = Faster processing BUT longer wait time in queue.

---

## 🎯 Quick Start Checklist

- [ ] SSH into HPC cluster
- [ ] Clone/upload project code
- [ ] Create conda environment (`conda create -n kiva_env python=3.10`)
- [ ] Install packages (`pip install torch transformers pillow pandas matplotlib`)
- [ ] Create logs directory (`mkdir -p logs`)
- [ ] Edit `anil.sbatch` with correct paths
- [ ] Submit job (`sbatch anil.sbatch`)
- [ ] Monitor progress (`tail -f logs/llava_test_*.out`)
- [ ] Check results (`cat output/multi_kiva/all_results.csv`)

---

## 📝 Notes for Beginners

1. **Start small**: Test with a few images first by modifying the loop range
2. **Use interactive sessions** for debugging before submitting jobs
3. **Save your work frequently** - the cluster might restart
4. **Check quotas**: Some clusters limit storage/compute time
5. **Be patient**: Large models take time to download and run
6. **Ask questions**: Cluster admins and lab mates are there to help!

---

## 👨‍💻 Author

**Anil Prajapati**

---

## 📄 License

This project is for academic/research purposes.

---

