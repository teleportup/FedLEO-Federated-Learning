# FedLEO - Installation & Usage Guide

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/teleportup/FedLEO-Federated-Learning.git
cd FedLEO-Federated-Learning
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run in Google Colab
- Open Google Colab: https://colab.research.google.com
- Import Notebook: Upload the `.ipynb` file
- Execute all cells

## Project Structure

```
FedLEO-Federated-Learning/
├── README.md                 # Main documentation
├── requirements.txt          # Python dependencies
├── FedLEO_notebook.ipynb     # Full Jupyter Notebook
└── README_INSTALLATION.md    # This file
```

## Features

✅ **Distributed Learning**: 4 virtual LEO satellites  
✅ **Weighted Aggregation**: Fair contribution based on data size  
✅ **Advanced Monitoring**: Real-time metrics tracking  
✅ **Visualization**: Convergence plots (Loss & Accuracy)  
✅ **Model Evaluation**: Complete test suite  

## Performance Metrics

- **Initial Accuracy**: 67.18%
- **Final Accuracy**: 85.19%
- **Loss Reduction**: 51.7%
- **Convergence**: 2 rounds

## System Requirements

- Python 3.8+
- 8GB+ RAM
- GPU recommended (NVIDIA CUDA)
- TensorFlow 2.19.0+

## Troubleshooting

For issues, check:
1. Python version: `python --version`
2. Dependencies: `pip list | grep tensorflow`
3. GPU availability: `nvidia-smi`

## License

MIT License - See LICENSE file
