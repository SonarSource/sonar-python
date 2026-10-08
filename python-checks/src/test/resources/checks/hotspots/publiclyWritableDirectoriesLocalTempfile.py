import tempfile
from tempfile import NamedTemporaryFile
from tempfile import TemporaryDirectory as temporary_directory


tempfile.TemporaryFile(dir="/tmp") # Noncompliant
NamedTemporaryFile(dir="/tmp") # Noncompliant
temporary_directory(dir="/tmp") # Noncompliant
tempfile.mkstemp(dir="/tmp") # Noncompliant
tempfile.mkdtemp(dir="/tmp") # Noncompliant
