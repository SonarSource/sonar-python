import tempfile
import os

filename = os.tempnam() # Noncompliant {{'os.tempnam' is insecure. Use 'tempfile.TemporaryFile' instead}}
#          ^^^^^^^^^^^^
tmp_file = open(filename, "w+b")

filename = os.tmpnam() # Noncompliant {{'os.tmpnam' is insecure. Use 'tempfile.TemporaryFile' instead}}
#          ^^^^^^^^^^^
tmp_file = open(filename, "w+b")

filename = tempfile.mktemp() # Noncompliant {{'tempfile.mktemp' is insecure. Use 'tempfile.TemporaryFile' instead}}
#          ^^^^^^^^^^^^^^^^^

tmp_file = open(filename, "w+b")


tmp_file = tempfile.TemporaryFile() # Compliant
