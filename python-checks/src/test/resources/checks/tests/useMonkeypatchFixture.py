import os

import myapp.config


def test_api_key():
    os.environ['API_KEY'] = 'test_key'
