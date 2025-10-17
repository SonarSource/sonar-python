To run:

in `python-frontend/src/main/protobuf` directory, run: ```protoc -I=. --python_out=../../../typeshed_serializer/serializer/proto_out ./symbols.proto```

```pip install -r requirements.txt```


## Rebuild only custom symbols

```bash
tox -e selective-serialize -- custom
```

## Run a custom test

```bash
tox -e py39 -- tests/runners/test_tox_runner.py
```
## Run one specific unit test

```bash
tox -e py39 -- tests/runners/test_tox_runner.py::ToxRunnerTest::test_dry_run_unchanged_checksums -v 
```

## Dry run of tox_runner

- Use python, not tox.
- Will show which calls to the `tox` module would have been triggered, depending on the checksums and file-system state.
- Will not perform any change.

```bash
python runners/tox_runner.py --dry_run true 
```

Can also be run in fail fast mode, to reflect behaviour in CI (where failStubGenerationFast is true)

```bash
python runners/tox_runner.py --dry_run true --fail_fast true  
```