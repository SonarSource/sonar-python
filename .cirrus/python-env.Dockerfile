FROM gcr.io/language-team/base:latest
USER root
RUN apt-get update && apt-get install -y python3-pip tox
RUN cd /usr/local/bin \
    && ln -s /usr/bin/python3 python \
    && ln -s /usr/bin/pip3 pip
USER sonarsource
