# Model Comparison Framework

## Overview
[Description]

## Directory Structure
[Map of important files/folders]

## Setup Instructions

### Prerequisites
- [Docker](https://www.docker.com/get-started) (required)
- GPU drivers (optional, for GPU acceleration)

### Installation

1. **Navigate to the classifier directory**
   ```bash
   cd path/to/classifier
   ```

2. **Build the Docker container**
   ```bash
   docker-compose build classifier-[cpu|gpu]
   ```
   
3. **Starting the container**
   ```bash
   docker-compose up classifier-[cpu|gpu]
   ```

> Replace `[cpu|gpu]` with either `cpu` (for CPU-only machines) or `gpu` (if you have NVIDIA GPU hardware)

#### Default

By default, this runs the "compare_models" module. To run other files, modify the last line in the Dockerfile.

### AWS Training Setup

If you're using AWS for training:

1. **Use the Deep Learning Base/PyTorch GPU AMI for easy NVIDIA drivers setup**

2. **For instances with instance store (SSD):**
   - Use the provided `mount_ssd.sh` script before building the docker image to format the temporary instance store SSD volume, and redirect Docker to use it. Run this script each time you start the instance.

3. **Automatic shutdown script:**
   - Use `auto_shutdown.sh` to automatically shut down the instance after training.
   - Run it in the background using tmux:
     ```bash
     tmux new -d "bash auto_shutdown.sh"
     ```
   - To reattach to the tmux session:
     ```bash
     tmux attach
     ```
