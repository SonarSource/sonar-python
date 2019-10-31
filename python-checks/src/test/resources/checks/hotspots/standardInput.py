import sys
from sys import stdin, __stdin__
import fileinput

def python_3_builtin():
    sys.stdin  # Noncompliant {{Make sure that reading the standard input is safe here.}}
#       ^^^^^

    sys.__stdin__  # Noncompliant

    stdin #Noncompliant
    __stdin__ #Noncompliant


    sys.stdin.read()  # Noncompliant
    sys.stdin.readline()  # Noncompliant
    sys.stdin.readlines()  # Noncompliant

    sys.__stdin__.read() # Noncompliant
    sys.stdin.seekable()  # Ok

    foo('stdin', sys.stdin) # Noncompliant
    data = sys.stdin.read(1) # Noncompliant

    input('What is your password?') # Noncompliant

def python_2_builtin():
    raw_input('What is your password?') # Noncompliant

def from_fileinput():
    fileinput.input() # Noncompliant
    fileinput.FileInput() # Noncompliant

    fileinput.input(['setup.py']) # OK
    fileinput.FileInput(['setup.py']) # OK




