# In-Context Learning Beyond Copying

### Environment Setup
```bash
conda env create -f environment.yml

pip install flash-attn==2.7.4.post1
```

## Replicating of Training
The instructions for replicating our training pipeline is below. However, you can easily modify the code to train arbitrary huggingface models!

We use the deduplicated Pile dataset from EleutherAI. To prepare the training data:

1. Follow the [Pythia dataset instructions](https://github.com/EleutherAI/pythia?tab=readme-ov-file#exploring-the-dataset)
2. Use the **deduplicated version** of the dataset
3. Generate `document.bin` and `document.idx` files

Launch training with:
```bash
python train_pythia.py --config configs/<config>.json --run_name <run_name>
```
Training was performed on 8× NVIDIA A100 80GB GPUs. The training data is also splitted into training & validation dataset by using the ```train_valid_split``` parameter in the configuration.

## Evaluation

### Model Conversion
Convert saved checkpoints to Hugging Face format:

```bash
python convert_to_hf.py --model_type <run_name>
```
The evaluation scripts in `metrics/` read the converted models from `models/<run_name>/step=<step>`.

### Running Evaluations
- Evaluation and plotting scripts are in the `metrics` folder.
- The ICL tasks in `dataset/icl_tasks` are from the [Function Vectors repository](https://github.com/ericwtodd/function_vectors).

## Pre-trained Models
Pre-trained models will be available on Hugging Face soon!

## Acknowledgements
We thank [EleutherAI](https://github.com/EleutherAI) for their open-source
[Pythia repository](https://github.com/EleutherAI/pythia), which is licensed
under the Apache License, Version 2.0.  
Our implementation reuses the file `mmap_dataset.py` from that project.

## License
This repository is released under the **MIT License** (see `LICENSE`).

The included file `mmap_dataset.py` remains under the
**Apache License, Version 2.0** (see `LICENSE_pythia.txt` for details).
