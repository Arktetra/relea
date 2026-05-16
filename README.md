# relea

Learning representation learning one step at a time.

## Setup

First, clone the repository:

```bash
git clone https://github.com/Arktetra/relea.git
```

Then, install the project using:

```bash
pip install -e .
```

## Training

The training script can be run with a config as follows:

```bash
python train.py --config /configs/{model}/{config}.yaml
```

Example:

```bash
python train.py --config /configs/vae/vae.yaml
```

Then, the samples generated at certain epochs during the training can be viewed in `work_dirs` directory in the project root directory.