# Commands

## Generating the binaries for the typeshed and custom stubs:

in `python-frontend/typeshed_serializer` directory, run:
```./build-with-docker```

if for any reason the checksum are not different and the re-generation of the binaries does not start,
remove the checksum file. It will be re-generated.

## Updating the protobuf schemas:

in `python-frontend/src/main/protobuf` directory, run: 
```protoc -I=. --python_out=../../../typeshed_serializer/serializer/proto_out ./symbols.proto```

