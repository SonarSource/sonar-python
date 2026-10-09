To run:

in `python-frontend/src/main/protobuf` directory, run: ```protoc -I=. --python_out=../../../typeshed_serializer/serializer/proto_out ./symbols.proto```

```uv sync```

## Run the serializer

Semantic model targets are configured in
`../src/main/resources/org/sonar/python/python-semantic-versions.json`, read by both the
serializer and Java analyzer. Keep the versions ordered from oldest to newest.
Adding a target also requires its Java `SemanticVersion` constant and regenerated descriptors.

```bash
uv run python runners/serializer_runner.py
```

## Run the test suite

```bash
uv run python -m pytest tests/
```

## Rebuild only custom symbols

```bash
uv run python -m utils.folder_manager custom && uv run python -m serializer.typeshed_serializer custom
```

## Run a custom test

```bash
uv run python -m pytest tests/runners/test_serializer_runner.py
```
## Run one specific unit test

```bash
uv run python -m pytest tests/runners/test_serializer_runner.py::RunnerTest::test_dry_run_unchanged_checksums -v
```

## Dry run of runner

- Will show which calls would have been triggered, depending on the checksums and file-system state.
- Will not perform any change.

```bash
uv run python runners/serializer_runner.py --dry_run true
```

Can also be run in fail fast mode, to reflect the checksum validation used by the QA workflow

```bash
uv run python runners/serializer_runner.py --dry_run true --fail_fast true
```
