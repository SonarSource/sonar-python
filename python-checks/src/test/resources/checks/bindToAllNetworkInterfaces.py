import uvicorn
from fastapi import FastAPI
from flask import Flask
from somewhere import my_host_name

def fast_api_tests():
    app_fastapi = FastAPI()

    uvicorn.run(app_fastapi, host="127.0.0.1")
    uvicorn.run(app_fastapi, port=8000)
    uvicorn.run(app_fastapi, host="127.0.0.1", port=8000)
    uvicorn.run(app_fastapi, host="localhost", port=8000)
    uvicorn.run(app_fastapi, host="192.168.1.100", port=8000)
    uvicorn.run(app_fastapi, host=my_host_name, port=8000)

    uvicorn.run(app_fastapi, host="0.0.0.0") # Noncompliant {{Avoid binding the application to all network interfaces.}}
#   ^^^^^^^^^^^

    uvicorn.run(app_fastapi, host="0.0.0.0", port=8000) # Noncompliant
    uvicorn.run(app_fastapi, host="0.0.0.0", port=8000, debug=True) # Noncompliant
    host_config = "0.0.0.0"
    uvicorn.run(app_fastapi, host=host_config, port=8000) # Noncompliant

def flask_tests():
    app_flask = Flask(__name__)

    app_flask.run(host='0.0.0.0', debug=True)  # Noncompliant {{Avoid binding the application to all network interfaces.}}
#   ^^^^^^^^^^^^^
    app_flask.run('0.0.0.0', 5000, True)  # Noncompliant
    app_flask.run(host='0.0.0.0', debug=False) # Noncompliant
    app_flask.run(host='0.0.0.0') # Noncompliant
    app_flask.run(host='127.0.0.1', debug=True)
    app_flask.run(host='', debug=True)
    app_flask.run('', debug=True)
    app_flask.run(None, debug=True)
    debug = True
    host = '0.0.0.0'
    app_flask.run(host=host, debug=debug) # Noncompliant
    other_host = '12'
    app_flask.run(other_host, debug=debug)
    app_flask.run() # Incorrect syntax
    reassigned = '0.0.0.0'
    app_flask.run(reassigned, debug=True) # FN limitation of singleAssignedValue
    reassigned = '0.0.0.0'
