from starlette.testclient import TestClient
from starlette.applications import Starlette

strVar = ""
bytesVar = b""
intVar = 1

app = Starlette()
client = TestClient(app)

response = client.post('/api', data=b'raw bytes')  # Noncompliant {{Use "content" parameter instead of "data" for bytes or text.}}
                            #  ^^^^

response = client.put('/api', data='text content')  # Noncompliant

response = client.get('/api', data=b'bytes')  # Noncompliant
response = client.delete('/api', data=b'bytes')  # Noncompliant
response = client.patch('/api', data=b'bytes')  # Noncompliant

response = client.request('POST', '/api', data=b'bytes')  # Noncompliant

response = client.post('/api', data='')  # Noncompliant

response = client.post('/api', data=b'')  # Noncompliant

response = client.post('/api', content=b'raw bytes')  # Compliant

response = client.put('/api', content='text content')  # Compliant

response = client.post('/api', data={'field': 'value'})  # Compliant

response = client.post('/api', json={'key': 'value'})  # Compliant

response = client.get('/api')  # Compliant

response = client.post('/api', data=None)  # Compliant

response = client.post('/api', data=strVar)  # Noncompliant
response = client.post('/api', data=bytesVar)  # Noncompliant
response = client.post('/api', data=intVar)  # Compliant

import requests
response = requests.post('/api', data=b'bytes')  # Compliant

response = TestClient(app).post('/api', data=b'bytes')  # Noncompliant

from fastapi import FastAPI
from fastapi.testclient import TestClient as FastAPITestClient

fastapi_app = FastAPI()
fastapi_client = FastAPITestClient(fastapi_app)

response = fastapi_client.post('/upload', data=b'\x89PNG\r\n')  # Noncompliant

response = fastapi_client.put('/api', data='text')  # Noncompliant

response = fastapi_client.post('/upload', content=b'\x89PNG\r\n')  # Compliant

response = fastapi_client.post('/api', data={'field': 'value'})  # Compliant

from starlette.testclient import TestClient as TC
aliased_client = TC(app)

response = aliased_client.post('/api', data=b'bytes')  # Noncompliant

name = "world"
response = client.post('/api', data=f'hello {name}')  # Noncompliant

response = client.post('/api', headers={'X-Header': 'value'}, data=b'bytes')  # Noncompliant

local_client = TestClient(app)
response = local_client.post('/api', data=b'bytes')  # Noncompliant

def get_some_data():
    return {}
unknown_variable = get_some_data()
response = client.post('/api', data=unknown_variable)  # Compliant

form_data = {'field': 'value'}
response = client.post('/api', data=form_data)  # Compliant
