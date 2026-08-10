# Non-pytest-named production module: test_* helpers with defaults must not be flagged
def test_connection(timeout=30):
    return timeout


class Client:
    def test_endpoint(self, retries=3):
        return retries
