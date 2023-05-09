import logging
import typing as t
from .app import Flask as Flask
from .globals import request as request
from typing import Any

def wsgi_errors_stream() -> t.TextIO: ...
def has_level_handler(logger: logging.Logger) -> bool: ...

default_handler: Any

def create_logger(app: Flask) -> logging.Logger: ...
