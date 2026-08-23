# POT-PTQ for Large Language Models

This repository reproduces two-step Power-of-Two post-training quantization
for causal language models and evaluates the result on WikiText-2.

## Recommended: Run on Kaggle

Use a Kaggle notebook with **Internet enabled** and a **GPU accelerator**.

### 1. Clone the repository

Run this in the first Kaggle cell:

```python
!git clone https://github.com/code-with-idrees/Power-of-Two-Quantization-Aware-Training-PoT-QAT-in-Large-Language-Models-LLMs-.git
%cd Power-of-Two-Quantization-Aware-Training-PoT-QAT-in-Large-Language-Models-LLMs-
```

### 2. Install dependencies

```python
!pip install -q -r requirements.txt
```

Restart the Kaggle session if Kaggle asks you to do so, then run the clone
and install cells again if needed.

### 3. Run the experiment

The default model is the public `TinyLlama/TinyLlama_v1.1`, so no token is
needed:

```python
!python run_pot_ptq_1b.py --eval_max_chunks 5
```

The `--eval_max_chunks 5` option is a quick test. For a longer evaluation:

```python
!python run_pot_ptq_1b.py
```

The script will load the model, prepare WikiText-2, evaluate the FP16
baseline, run 3-bit and 2-bit POT-PTQ, and print perplexity results.

### 4. Lower-memory quick run

If the GPU runs out of memory or the experiment is too slow, use smaller
calibration settings:

```python
!python run_pot_ptq_1b.py --calib_seqs 8 --calib_seqlen 512 --eval_max_chunks 2 --epochs_3bit 1 --epochs_2bit 1
```

To run only 3-bit quantization:

```python
!python run_pot_ptq_1b.py --skip_bits 2
```

## Using Llama 3.2 1B

`meta-llama/Llama-3.2-1B` is gated. First accept its license on Hugging
Face, create an access token, and add it to Kaggle as a secret named
`HF_TOKEN`. Then run:

```python
import os
os.environ["HF_TOKEN"] = "your_token_from_kaggle_secret"
!python run_pot_ptq_1b.py --model_name meta-llama/Llama-3.2-1B
```

Do not publish the token in the notebook or repository.

## Optional experiments

```python
!python run_pot_ptq_1b.py --run_ablation --run_benchmark --eval_max_chunks 5
```

The optional flags compare initialization methods and benchmark the two
dequantization implementations.

## Repository layout

```text
run_pot_ptq_1b.py       Main command-line runner
pot_ptq/                Quantization and evaluation package
test_pot_ptq.py         Unit tests
requirements.txt        Python dependencies
POT_PTQ_Llama_1B.ipynb  Optional notebook workflow
```

The notebooks are optional. The runner and `pot_ptq` folder are the required
files for the Kaggle command-line workflow.

## Run locally

```bash
pip install -r requirements.txt
python run_pot_ptq_1b.py --model_name TinyLlama/TinyLlama_v1.1
```

Run the tests with:

```bash
python -m unittest test_pot_ptq.py
```
