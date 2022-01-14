#base image
FROM gcr.io/deeplearning-platform-release/pytorch-gpu
RUN apt update && \
   apt install --no-install-recommends -y build-essential gcc wget curl python3.9 && \
   apt clean && rm -rf /var/lib/apt/lists/*


RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /root

COPY requirements.txt /tmp/requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY .git/ .git/
COPY .dvc/config .dvc/config
COPY .dvc/plots .dvc/plots
COPY config/ config/
COPY data/processed.dvc data/processed.dvc

RUN python3.9 -m pip install -r /tmp/requirements.txt --no-cache-dir
RUN pip install dvc 'dvc[gs]'

RUN wget -nv \
    https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
    mkdir /root/tools && \
    tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
    rm google-cloud-sdk.tar.gz && \
    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
        --path-update=false --bash-completion=false \
        --disable-installation-options && \
    rm -rf /root/.config/* && \
    ln -s /root/.config /config && \
    # Remove the backup directory that gcloud creates
    rm -rf /root/tools/google-cloud-sdk/.install/.backup

# Path configuration
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin
# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

COPY docker_run_training.sh docker_run_training.sh
ENTRYPOINT ["./docker_run_training.sh"]
