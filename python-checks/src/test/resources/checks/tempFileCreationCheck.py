import tempfile
import os
from os import tempnam
from os import tmpnam

filename = os.tempnam() # Noncompliant {{'os.tempnam' is insecure. Use 'tempfile.TemporaryFile' instead}}
#          ^^^^^^^^^^
tmp_file = open(filename, "w+b")

filename = os.tmpnam() # Noncompliant {{'os.tmpnam' is insecure. Use 'tempfile.TemporaryFile' instead}}
#          ^^^^^^^^^
tmp_file = open(filename, "w+b")

filename = tempfile.mktemp() # Noncompliant {{'tempfile.mktemp' is insecure. Use 'tempfile.TemporaryFile' instead}}
#          ^^^^^^^^^^^^^^^

tmp_file = open(filename, "w+b")

filename = tempnam() # Noncompliant {{'os.tempnam' is insecure. Use 'tempfile.TemporaryFile' instead}}
#          ^^^^^^^

filename = tmpnam() # Noncompliant {{'os.tmpnam' is insecure. Use 'tempfile.TemporaryFile' instead}}
#          ^^^^^^

tmp_file = tempfile.TemporaryFile() # Compliant
