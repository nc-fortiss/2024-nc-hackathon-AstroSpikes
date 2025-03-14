# SNN MobileNet Implementation

## Python environment setup instructions

```bash

< conda or pip env with python=3.10 >

<activate environment>

wget https://pypi.nvidia.com/tensorrt-libs/tensorrt_libs-8.6.1-py2.py3-none-manylinux_2_17_x86_64.whl#sha256=b8445cdba68d108345c95a65167c2bb6e03cb3e6cd6cb51e86f1028151d5a93e

pip install --no-deps ./tensorrt_libs-8.6.1-py2.py3-none-manylinux_2_17_x86_64.whl

pip install tensorflow[and-cuda]==2.15

pip install akida_models==1.6.1 wandb==0.18.0 tonic==1.5.0 omegaconf==2.3.0 tensorflow-addons==0.23.0

```

# Renaming Dataset

dataset sequences are renamed for convenience

```markdown
seq_RT000 -> RT000
```

labels 

```markdown
seq_RT000.csv -> RT000.csv
```

this can be done through [the renaming script](./scripts/rename_SPADES.py).


# Training

- add the wandb api key

```bash
python train.py
```


