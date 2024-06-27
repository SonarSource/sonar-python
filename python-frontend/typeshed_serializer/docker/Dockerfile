FROM python:3.9

# Add typeshed serializer to the image container
COPY --chmod=777 ./ /sonar-python/python-frontend/typeshed_serializer
WORKDIR /sonar-python/python-frontend/typeshed_serializer

# Install tox
RUN pip install --no-cache-dir tox

# Initializetion of the tox environments
RUN python -m tox --recreate --notest

# Grant premissions to read files
RUN chmod -R 777 ./

# Make run script executable
RUN chmod +x ./docker/cmd.sh

# Run runners/tox_runner.py when the container launches
CMD ["./docker/cmd.sh"]
