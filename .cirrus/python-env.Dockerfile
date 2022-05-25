FROM eu.gcr.io/release-engineering-ci-prod/base:j11-latest
USER root
ENV PYTHON_VERSION=3.9.5
RUN apt-get update && apt-get install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libsqlite3-dev libreadline-dev libffi-dev curl libbz2-dev
RUN curl -O https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tar.xz
RUN tar -xf Python-${PYTHON_VERSION}.tar.xz
RUN cd Python-${PYTHON_VERSION} && ./configure --enable-optimizations && make -j 4 && make altinstall
RUN cd /usr/local/bin \
    && ln -s python3.9 python \
    && ln -s python3.9 python3 \
    && ln -s pip3.9 pip \
    && ln -s pip3.9 pip3
RUN python3.9 -m pip install --upgrade pip
USER sonarsource
RUN pip install tox
ENV PATH=${PATH}:/home/sonarsource/.local/bin