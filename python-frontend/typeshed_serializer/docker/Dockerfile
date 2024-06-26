FROM python:3.9
ARG UID
ARG GID


# Add typeshed serializer to the image container
ADD --chown=${UID}:${GID} ./ /sonar-python/python-frontend/typeshed_serializer
WORKDIR /sonar-python/python-frontend/typeshed_serializer

# Create user with same IDs as host machine current user
# Switch to host user
RUN groupadd -g ${GID} -o host_group
RUN useradd -m -u ${UID} -g ${GID} -o -s /bin/bash host_user
USER host_user

# Install tox
RUN pip install --no-cache-dir tox

# Initializetion of the tox environments
RUN python -m tox --recreate --notest

# Make run script executable
RUN chmod +x ./docker/cmd.sh

# Run runners/tox_runner.py when the container launches
CMD ["./docker/cmd.sh"]
