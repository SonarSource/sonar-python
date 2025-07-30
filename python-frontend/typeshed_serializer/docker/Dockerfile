FROM python:3.9

RUN apt-get update && \
    apt-get install -y protobuf-compiler && \
    rm -rf /var/lib/apt/lists/*

# Verify that protoc is installed and accessible on the PATH
# This command will print the version of protoc, confirming its installation
RUN protoc --version

# Install tox
RUN pip install --no-cache-dir tox

# Add typeshed serializer to the image container
COPY --chmod=777 ./docker/cmd.sh /sonar-python/cmd.sh

# Make run script executable
RUN chmod +x /sonar-python/cmd.sh

# Run runners/tox_runner.py when the container launches
CMD ["/sonar-python/cmd.sh"]
