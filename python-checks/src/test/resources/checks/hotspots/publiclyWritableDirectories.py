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

def environ_variables():
    import os
    import myos
    from os import environ
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
