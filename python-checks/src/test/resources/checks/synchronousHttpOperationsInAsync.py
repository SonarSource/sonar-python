

# Test cases for requests library
async def test_requests_sync_client():
    import requests

    response = requests.get("https://example.com")  # Noncompliant

    response = requests.post("https://example.com", data={"key": "value"})  # Noncompliant

    session = requests.Session()
    response = session.get("https://example.com")  # Noncompliant

    # Methods test
    requests.delete("https://example.com")  # Noncompliant
    requests.head("https://example.com")  # Noncompliant
    requests.options("https://example.com")  # Noncompliant
    requests.patch("https://example.com")  # Noncompliant
    requests.put("https://example.com")  # Noncompliant

async def nesting():
    import requests
    async def inner_function():
#   ^^^^^> {{This function is async.}}
        requests.get("https://example.com")  # Noncompliant {{Use an async HTTP client in this async function instead of a synchronous one.}}
    #   ^^^^^^^^^^^^

# Test cases for urllib3 library
async def test_urllib3_sync_client():
    import urllib3

    http = urllib3.PoolManager() # Noncompliant
    response = http.request("GET", "https://example.com")  # Noncompliant

    response = http.request_encode_url("GET", "https://example.com")  # FN

    response = http.request_encode_body("POST", "https://example.com", fields={"key": "value"})  # FN

# Test cases for httpx synchronous client
async def test_httpx_sync_client():
    import httpx

    response = httpx.get("https://example.com")  # Noncompliant

    client = httpx.Client()
    response = client.get("https://example.com")  # FN SONARPY-2965

    with httpx.Client() as client:
        response = client.get("https://example.com")  # FN SONARPY-2965

# Test compliant cases with async HTTP clients
async def test_compliant_async_clients():
    # httpx async client
    import httpx

    async with httpx.AsyncClient() as client:
        response = await client.get("https://example.com")

    client = httpx.AsyncClient()
    response = await client.get("https://example.com")

    # aiohttp client
    import aiohttp

    async with aiohttp.ClientSession() as session:
        async with session.get("https://example.com") as response:
            data = await response.text()

    # asks library
    import asks

    response = await asks.get("https://example.com")

# Test nested async functions
async def test_nested_async_functions():
    import requests

    async def inner_function():
        return requests.get("https://example.com")  # Noncompliant

    result = await inner_function()

# Test functions that use synchronous HTTP clients but are not async themselves
def test_sync_function_with_sync_client():
    import requests

    response = requests.get("https://example.com")  # Compliant - not in async function

# Test for method calls on returned objects
async def test_method_chaining():
    import requests

    data = requests.get("https://example.com").json()  # Noncompliant

    import urllib3
    http = urllib3.PoolManager() # Noncompliant
    data = http.request("GET", "https://example.com").data  # Noncompliant

# Test for assignments to variables and complex expressions
async def test_complex_assignments():
    import requests

    # Assignment to variable
    response = requests.get("https://example.com")  # Noncompliant

    # In a complex expression
    data = [requests.get(f"https://example.com/{i}").json() for i in range(5)]  # Noncompliant

    # As function argument
    process_response(requests.get("https://example.com"))  # Noncompliant

# Test for HTTP client in async generators
async def test_async_generator():
    import requests

    async def async_generator():
        for i in range(5):
            yield requests.get(f"https://example.com/{i}")  # Noncompliant

    async for response in async_generator():
        print(response.text)

# Test for HTTP clients created outside async function but used inside
async def test_client_created_outside():
    import httpx
    client = httpx.Client()

    response = client.get("https://example.com")  # FN

    # Compliant case with async client
    async_client = httpx.AsyncClient()
    response = await async_client.get("https://example.com")

# Test for conditional usage of HTTP clients
async def test_conditional_usage():
    import requests
    import httpx

    use_async = False

    if use_async:
        async with httpx.AsyncClient() as client:
            response = await client.get("https://example.com")
    else:
        response = requests.get("https://example.com")  # Noncompliant

# Test for try/except blocks
async def test_try_except():
    import requests

    try:
        response = requests.get("https://example.com")  # Noncompliant
    except Exception:
        pass

# Edge cases to ensure no false positives
async def test_edge_cases():
    # String containing 'requests.get' should not trigger
    code_string = "response = requests.get('https://example.com')"

    # Variable named 'requests' should not trigger if it's not the actual requests module
    class MockRequests:
        def get(self, url):
            return None

    requests = MockRequests()
    response = requests.get("https://example.com")  # Should not trigger
