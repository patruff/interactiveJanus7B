# Interactive Janus-Pro-7B Image Generator

A Jupyter notebook for interactive image generation using Janus-Pro-7B. So run the cell to clone their repo (see the .ipynb file) and then just copy and paste the contents of the .py file in the next cell of the Colab and make sure to have your HF_TOKEN set as a secret in your Colab notebook. Enjoy!

## Requirements
- A100 GPU (40GB or 80GB)
- Python >= 3.8
- Google Colab with A100

## Setup
1. Clone Janus:
```bash
!git clone https://github.com/deepseek-ai/Janus.git
%cd Janus
!pip install -e .
```

2. Run the notebook

## Usage
1. Specify number of images per prompt
2. Enter text descriptions
3. Find generated images in `generated_samples` folder

## Output
Generated images are stored in `generated_samples` with timestamped filenames.

## Credits
Uses [Janus model](https://github.com/deepseek-ai/Janus) by DeepSeek AI
