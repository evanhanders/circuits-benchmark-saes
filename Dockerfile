ARG PYTORCH_CUDA_VERSION=2.3.1-cuda11.8-cudnn8
FROM pytorch/pytorch:${PYTORCH_CUDA_VERSION}-runtime as main-pre-pip

ENV DEBIAN_FRONTEND=noninteractive

# Install some useful packages
RUN apt-get update -q \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
    # essential for running
    git git-lfs tini \
    # additional packages for circuits bench
    libgl1-mesa-glx graphviz graphviz-dev build-essential \
    # nice to have for devbox development
    curl vim tmux less sudo rsync wget \
    # CircleCI
    ssh \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ARG USERID=1001
ARG GROUPID=1001
ARG USERNAME=dev

# Simulate virtualenv activation
ENV VIRTUAL_ENV="/opt/venv"
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

RUN python3 -m venv "${VIRTUAL_ENV}" --system-site-packages \
    && addgroup --gid ${GROUPID} ${USERNAME} \
    && adduser --uid ${USERID} --gid ${GROUPID} --disabled-password --gecos '' ${USERNAME} \
    && usermod -aG sudo ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
    && mkdir -p "/workspace" \
    && chown -R ${USERNAME}:${USERNAME} "${VIRTUAL_ENV}" "/workspace"
USER ${USERNAME}
WORKDIR "/workspace"

## Set PATH and install Jupyter
RUN /bin/bash -c 'mkdir -p $HOME && touch $HOME/.profile && echo "export PATH=\$PATH:\$HOME/.local/bin" >> $HOME/.profile'
RUN /bin/bash -c 'source $HOME/.profile && pip install -U jupyter'

# Copy package installation instructions and version.txt files
COPY --chown=${USERNAME}:${USERNAME} pyproject.toml ./

# Install content-less packages and their dependencies
RUN mkdir project_template \
    && touch project_template/__init__.py \
    && pip install --require-virtualenv --config-settings editable_mode=compat -e '.[dev]' \
    && rm -rf "${HOME}/.cache" "./dist" \
    # Uninstall tracr if it got installed for compatibility with circuits-benchmark
    && pip uninstall -y tracr \
    # Run Pyright so its Node.js package gets installed
    && pyright .

# Clone the repository and install
RUN git clone --recurse-submodules https://github.com/evanhanders/circuits-benchmark.git \
    && cd circuits-benchmark \
    && pip install -q -e .

# Copy whole repo
COPY --chown=${USERNAME}:${USERNAME} . .

# Default command to run -- may be changed at runtime
CMD ["/bin/bash"]
