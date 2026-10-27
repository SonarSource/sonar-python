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

## Third-party library coverage

Use one library version per distribution, favoring the most-used supported series.
Pin typing sources in `pyproject.toml` and keep `uv.lock` synchronized.
Prefer newer releases when choosing between commonly used versions. Signature-based
rules may report false positives or miss issues on older library versions; document
representative limitations in existing rule fixtures instead of modeling every release.

Review name resolution separately from signature compatibility, particularly for
security rules. For this refresh, SQLAlchemy 1.x's `Engine.execute` and pyOpenSSL's
removed `load_pkcs12` no longer resolve; their current replacements and existing
OpenSSL security findings are covered by tests. This does not establish security
coverage for applications using the removed APIs.

The importer explicitly loads public modules from installed distributions. Installed
inline typing and stub packages are selected before Typeshed's third-party fallback;
at analysis time imported descriptors take precedence over Typeshed descriptors.
Custom modules take precedence over both, so remove obsolete custom declarations
alongside their complete replacements.
The `chardet.universaldetector` override only restores its supported re-export of
`chardet.detector.UniversalDetector`; the package root and class stay upstream.

Validate public aliases, returned instances and rule diagnostics, including valid-call
controls. Successful mypy serialization alone does not establish analysis coverage.
The third-party source refresh is independent of the standard-library Typeshed pin;
regeneration must leave its stdlib descriptors unchanged.
