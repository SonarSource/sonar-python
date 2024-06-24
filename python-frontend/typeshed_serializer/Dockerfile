FROM python:3.9
ARG UID
ARG GID
RUN groupadd -g ${GID} -o group
RUN useradd -m -u ${UID} -g ${GID} -o -s /bin/bash user
RUN pip install --no-cache-dir tox
ADD . /sonar-python/python-frontend/typeshed_serializer
WORKDIR /sonar-python/python-frontend/typeshed_serializer
RUN tox --recreate --notest
RUN chmod +x docker-script.sh
RUN chown -R ${UID}:${GID} ./
# Run runners/tox_runner.py when the container launches
CMD ["./docker-script.sh"]