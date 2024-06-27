FROM python:3.9

# Add typeshed serializer to the image container
COPY --chmod=666 ./ /sonar-python/python-frontend/typeshed_serializer
# ADD --chown=${UID}:${GID} ./ /sonar-python/python-frontend/typeshed_serializer
WORKDIR /sonar-python/python-frontend/typeshed_serializer

# Install tox
RUN pip install --no-cache-dir tox

# Initializetion of the tox environments
RUN python -m tox --recreate --notest

# Make run script executable
RUN chmod +x ./docker/cmd.sh

# Grant premissions
RUN chmod 666 -R ./

# Run runners/tox_runner.py when the container launches
CMD ["./docker/cmd.sh"]
