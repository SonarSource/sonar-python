To run:

in `python-frontend/src/main/protobuf` directory, run: 
```protoc -I=. --python_out=../../../typeshed_serializer/serializer/proto_out ./symbols.proto --experimental_allow_proto3_optional```

```pip install -r requirements.txt```
