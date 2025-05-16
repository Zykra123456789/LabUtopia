# LabUtopia: High-Fidelity Simulation and Hierarchical Benchmark for Scientific Embodied Agents

This repository contains code for "LabUtopia: High-Fidelity Simulation and Hierarchical Benchmark for Scientific Embodied Agents".

## System Requirements
- NVIDIA GPU with CUDA support
- Python 3.10
- Isaac Sim 4.2

## 🛠️ Installation

### 1. Environment Creation
Create and activate a new conda environment:
```bash
conda create -n labutopia python=3.10
conda activate labutopia
```

### 2. Dependencies Installation
Install required packages:
```bash
# Install PyTorch
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install Isaac Sim
pip install isaacsim==4.2.0.2 isaacsim-extscache-physics==4.2.0.2 isaacsim-extscache-kit==4.2.0.2 isaacsim-extscache-kit-sdk==4.2.0.2 --extra-index-url https://pypi.nvidia.com

# Install other dependencies
pip install -r requirements.txt
```

## Usage

### Data Collection
To collect training data:
1. Modify the configuration file in `config/`
2. Run:
```bash
python main.py --config-name level1_pick
```

### Training
To train the model:
1. Adjust parameters in `policy/config/`
2. Run:
```bash
python train.py --config-name=train_diffusion_unet_image_workspace
```

### Inference
To run inference with a trained model:
1. Update the model path in the config file, change the 'mode' to 'infer'.
2. Run:
```bash
python main.py --config-name level1_pick
```