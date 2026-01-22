import uvicorn
from fastapi import FastAPI
from somewhere import my_host_name

app = FastAPI()

uvicorn.run(app, host="127.0.0.1")
uvicorn.run(app, port=8000)
uvicorn.run(app, host="127.0.0.1", port=8000)
uvicorn.run(app, host="localhost", port=8000)
uvicorn.run(app, host="192.168.1.100", port=8000)
uvicorn.run(app, host=my_host_name, port=8000)

uvicorn.run(app, host="0.0.0.0") # Noncompliant {{Avoid binding the FastAPI application to all network interfaces.}}
uvicorn.run(app, host="0.0.0.0", port=8000) # Noncompliant
uvicorn.run(app, host="0.0.0.0", port=8000, debug=True) # Noncompliant
host_config = "0.0.0.0"
uvicorn.run(app, host=host_config, port=8000) # Noncompliant
