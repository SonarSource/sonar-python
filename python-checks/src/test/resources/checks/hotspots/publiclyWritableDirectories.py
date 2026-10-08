# Common
open("/tmp/f","w+") # Noncompliant
open("/tmp","w+") # Noncompliant
open("/var/tmp/f","w+") # Noncompliant
open("/usr/tmp/f","w+") # Noncompliant
open("/dev/shm/f","w+") # Noncompliant
open("dev/shm/f","w+") # OK
open("/tmpx","w+") # OK
open() # OK

# Linux
open("/dev/mqueue/f","w+") # Noncompliant
open("/run/lock/f","w+") # Noncompliant
open("/var/run/lock/f","w+") # Noncompliant

# MacOS
open("/Library/Caches/f","w+") # Noncompliant
open("/Users/Shared/f","w+") # Noncompliant
open("/private/tmp/f","w+") # Noncompliant
open("/private/var/tmp/f","w+") # Noncompliant

# Windows
open(r"\Windows\Temp\f") # Noncompliant
open(r"D:\Windows\Temp\f") # Noncompliant
open(r"\Windows\Temp\f") # Noncompliant
open(r"\Temp\f") # Noncompliant
open(r"\TEMP\f") # Noncompliant
open(r"\TMP\f") # Noncompliant
open(r"C:\Temperatures") # OK

import tempfile
from tempfile import NamedTemporaryFile as named_temporary_file
from tempfile import TemporaryDirectory as temporary_directory
from tempfile import TemporaryFile

tempfile.TemporaryFile(dir="/tmp") # OK
tempfile.TemporaryFile(dir=("/tmp")) # OK
tempfile.NamedTemporaryFile(dir="/var/tmp") # OK
tempfile.TemporaryDirectory(dir="/dev/shm") # OK
tempfile.SpooledTemporaryFile(dir="/tmp") # OK
tempfile.mkstemp(dir="/var/tmp") # OK
tempfile.mkdtemp(dir="/dev/shm") # OK
tempfile.TemporaryFile(prefix="/tmp") # Noncompliant
TemporaryFile(dir="/tmp") # OK
named_temporary_file(dir="/var/tmp") # OK
temporary_directory(dir="/dev/shm") # OK

tempfile.TemporaryFile(dir=r"C:\Windows\Temp") # OK
tempfile.NamedTemporaryFile(dir="C:\\Temp") # OK
tempfile.TemporaryDirectory(dir=r"D:\TMP") # OK
tempfile.SpooledTemporaryFile(dir=r"\Temp") # OK
tempfile.mkstemp(dir="C:\\Windows\\Temp\\") # OK
tempfile.mkdtemp(dir=r"D:\tEmP") # OK
named_temporary_file(dir=r"C:\Temp") # OK
temporary_directory(dir=("C:\\Windows\\Temp")) # OK

tempfile.mktemp(dir=r"C:\Temp") # Noncompliant
tempfile.NamedTemporaryFile(prefix=r"C:\Temp\report") # Noncompliant


# Literal directory arguments, including the compliant example from S5443.
tempfile.TemporaryFile(dir="/tmp/my_subdirectory", mode="w+") # OK
tempfile.NamedTemporaryFile(dir="/tmp/myapp") # OK
tempfile.mkdtemp(dir=r"C:\Temp\app") # OK
tempfile.mkdtemp(dir=r"app\Temp") # OK
tempfile.mkdtemp(dir=r"C:app\Temp") # OK
tempfile.mkstemp(dir="/tmp/") # OK
tempfile.mkstemp(dir="/tmp" "/") # OK
tempfile.mkstemp(dir="/tmp" "/myapp") # OK
tempfile.mkstemp(dir="/tmp" if cond else ("/var/tmp")) # OK
tempfile.NamedTemporaryFile(dir=(("/tmp") if cond else ("/var/tmp" if other else "/dev/shm"))) # OK

# The dir parameter can also be passed by position.
tempfile.mkstemp(".txt", "report", "/tmp") # OK
tempfile.mkdtemp(None, None, "/var/tmp") # OK
tempfile.TemporaryDirectory(None, None, "/dev/shm") # OK
tempfile.TemporaryFile("w+b", -1, None, None, None, None, "/tmp") # OK
tempfile.NamedTemporaryFile("w+b", -1, None, None, None, None, r"C:\Temp") # OK
tempfile.SpooledTemporaryFile(0, "w+b", -1, None, None, None, None, "/tmp") # OK
named_temporary_file("w+b", -1, None, None, None, None, "/var/tmp") # OK
tempfile.mkstemp(".txt", "report", "/tmp" if cond else "/var/tmp") # OK

# A preceding unpacked argument prevents reliable positional matching, but not keyword matching.
tempfile.mkstemp(*(), ".txt", "report", dir="/tmp") # OK
tempfile.mkstemp(*args, dir="/tmp" if cond else "/var/tmp") # OK
tempfile.mkstemp(*(), ".txt", "/tmp") # Noncompliant
tempfile.mkstemp(*(), ".txt", "report", "/tmp") # Noncompliant

# Other parameters, conditions, and computed values retain their findings.
tempfile.mkstemp("/tmp") # Noncompliant
tempfile.SpooledTemporaryFile(0, "w+b", -1, None, None, None, "/tmp") # Noncompliant
tempfile.mktemp(dir="/tmp") # Noncompliant
tempfile.NamedTemporaryFile(prefix="/tmp" if cond else "report") # Noncompliant
tempfile.mkstemp(dir=cond if "/tmp" else None) # Noncompliant
tempfile.mkstemp(dir=("/tmp" if cond else "/home/user") + "") # Noncompliant
tempfile.mkstemp(dir="/tmp" + "") # Noncompliant
class TempConfig(dir="/tmp"): # Noncompliant
    pass

def local_temporary_file():
    def TemporaryFile(dir):
        pass

    TemporaryFile(dir="/tmp") # Noncompliant

def environ_variables():
    import os
    import myos
    from os import environ
    tempfile.TemporaryFile(dir=os.environ.get('TMPDIR')) # Noncompliant
    tempfile.TemporaryDirectory(dir=os.environ['TMP']) # Noncompliant
    tmp_dir = os.environ.get('TMPDIR') # Noncompliant
    tmp_dir = os.environ.get('TMP') # Noncompliant
    tmp_dir = os.environ['TMPDIR'] # Noncompliant
    tmp_dir = os.environ[foo] # OK
    tmp_dir = os.environ.other_method('TMPDIR') # OK
    tmp_dir = os.environ.get('OTHER') # OK
    tmp_dir = os.environ['OTHER'] # OK
    tmp_dir = os.environ['OTHER'] # OK
    tmp_dir = os.other['TMPDIR'] # OK
    tmp_dir = other['TMPDIR'] # OK
    tmp_dir = foo()['TMPDIR'] # OK
    tmp_dir = os.foo.environ['TMPDIR'] # OK
    tmp_dir = environ['TMPDIR'] # Noncompliant
    tmp_dir = environ.get('TMPDIR') # Noncompliant
    tmp_dir = myos.environ.get('TMPDIR') # OK
