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

## Python 3.15 data

The serializer targets Python 3.10 through 3.15 while running on Python 3.14.
Both direct sources and transitive standard-library imports use the pinned Typeshed
revision. Module availability follows Typeshed's `stdlib/VERSIONS`; symbol and
member availability follows its version-conditional declarations.

The Typeshed pin `c333d5d346a469c49b309ddbafe1d8173fae7a21` includes Python 3.15
declarations and remains compatible with mypy 1.19.1. Its successor removes the
`builtins.ellipsis` pseudo-type still required by that mypy version.

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
The third-party source refresh is independent of the standard-library Typeshed pin.
This upgrade retains its selected versions and adds Stripe as a replacement for the
legacy declarations removed by the new Typeshed pin.

### Protobuf: 7.x only

The Python 3.15 Typeshed upgrade deliberately adopts the upstream Protobuf 7.x
API. There are no custom Protobuf overrides or historical-signature overloads.
The analyzer does not select stubs by a project's installed library version.
This decision concerns the Python library being analyzed, independently of the
Protobuf runtime used to serialize analyzer descriptors and caches.

Consequently, S930 can report unexpected arguments on valid older code:

- Protobuf 4.x accepts `including_default_value_fields` in `MessageToDict` and
  `MessageToJson`; Protobuf 5.x replaced it with
  `always_print_fields_with_no_presence`.
- Protobuf 6.x still accepts `float_precision` in the JSON formatters and
  `float_format` / `double_format` in the text formatters; Protobuf 7.x removed
  these options.

These are accepted false positives for unsupported dependency versions. The
examples in `argumentNumberDependencyVersions.py` document the limitation while
checking that valid Protobuf 7.x calls remain accepted. Revisit narrowly scoped
compatibility if user reports establish that the noise warrants an exception.

S5655 can also flag a valid Protobuf 6.x positional `MessageToJson` call: its
seventh argument was `float_precision` (`int | None`), but Protobuf 7.x interprets
that position as `ensure_ascii` (`bool`). The `None` precision case and valid 7.x
controls are documented in `argumentType.py`.

The TensorFlow ruling checkout uses older Protobuf semantics. Its calls in
`python/platform/benchmark_test.py` (`including_default_value_fields`),
`python/framework/graph_io.py`, and
`python/kernel_tests/proto/{encode,decode}_proto_op_test_base.py` (`float_format`)
are known S930 false positives under this baseline. Keep these findings visible
in the ruling expectations until the source checkout is updated; do not change
the analyzer signatures just to keep old fixtures quiet.

Upstream API changes:

- https://protobuf.dev/news/2024-02-05/ (default-value JSON option)
- https://protobuf.dev/news/2025-09-19/ (float options removed in Python 7.x)
