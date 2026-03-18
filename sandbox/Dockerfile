FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install common dev tools
RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    nodejs \
    npm \
    gcc \
    g++ \
    make \
    curl \
    wget \
    vim \
    jq \
    tree \
    htop \
    net-tools \
    iproute2 \
    procps \
    coreutils \
    findutils \
    grep \
    sed \
    gawk \
    diffutils \
    file \
    less \
    man-db \
    zip \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Create a realistic user environment
RUN useradd -ms /bin/bash developer
USER developer
WORKDIR /home/developer

# Copy workspace setup script
COPY --chown=developer:developer setup_workspace.sh /home/developer/setup_workspace.sh
RUN chmod +x /home/developer/setup_workspace.sh && /home/developer/setup_workspace.sh

# Default shell
CMD ["/bin/bash"]
