# Non-compliant code examples - compression modules imported directly (Python 3.14+)
from lzma import LZMAFile  # Noncompliant {{Compression modules should be imported from the compression namespace.}}
#    ^^^^
from bz2 import BZ2File  # Noncompliant
from gzip import GzipFile  # Noncompliant
from zlib import compress  # Noncompliant 

# Compliant code examples - compression modules imported from namespace
from compression.lzma import LZMAFile
from compression.bz2 import BZ2File
from compression.gzip import GzipFile
from compression.zlib import compress

# Other imports that should not trigger issues
from os import path
from sys import argv
from json import loads, dumps
from collections import defaultdict
from typing import List, Dict

# Multiple imports from compression modules
from lzma import LZMAFile, open, compress as lzma_compress  # Noncompliant 
from bz2 import open as bz2_open, BZ2File as BZ2  # Noncompliant

# Valid imports from other modules with similar names
from my_custom_lzma import CustomLZMA
from package.bz2_like import SomethingElse
from utils.gzip_helper import helper_function

# Import statements - Non-compliant (should use compression namespace)
import lzma  # Noncompliant
import bz2  # Noncompliant
import gzip  # Noncompliant
import zlib  # Noncompliant

# Multiple imports in one statement
import lzma, bz2, os  # Noncompliant 2

# Import with alias
import lzma as compression_lzma  # Noncompliant
import bz2 as bz  # Noncompliant

# Compliant import statements - using compression namespace
import compression.lzma
import compression.bz2
import compression.gzip
import compression.zlib

# Other valid imports that should not trigger issues
import os
import sys
import json
