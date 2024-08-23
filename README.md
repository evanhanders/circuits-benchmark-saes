# Installation

To install this repository, create a fresh poetry virtual environment. If you haven't used poetry before, it can be helpful to make sure the following poetry config options are set:

```sh
poetry config virtualenvs.create true
poetry config virtualenvs.in-project true
```

You can check these settings with 

```sh
poetry config --list
```

Then clone this repository and install it and its dependencies:
```sh
git clone https://github.com/evanhanders/polysemantic-benchmark.git
cd polysemantic-benchmark
poetry install
```

Once the repo is installed, you should be able to run one of the notebooks in the root folder to train or load models.
