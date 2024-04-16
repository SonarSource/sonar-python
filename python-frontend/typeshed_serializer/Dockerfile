FROM python:3.9

RUN pip install --no-cache-dir tox
ADD . /sonar-python/python-frontend/typeshed_serializer
WORKDIR /sonar-python/python-frontend/typeshed_serializer

# Run runners/tox_runner.py when the container launches
CMD ["python", "runners/tox_runner.py", "--skip_tests=true"]