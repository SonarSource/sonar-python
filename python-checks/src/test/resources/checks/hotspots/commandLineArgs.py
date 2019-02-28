import argparse
from argparse import ArgumentParser

import sys
from sys import argv
import mySys

from optparse import OptionParser
import optparse

def argparse_test():
    argparse.ArgumentParser() # Noncompliant {{Make sure that command line arguments are used safely here.}}
#   ^^^^^^^^^^^^^^^^^^^^^^^^^
    ArgumentParser() # Noncompliant
    argparse.otherFunction() # OK

def builtins():
    sys.argv # Noncompliant
#   ^^^^^^^^
    argv # FN, we only resolve qualified name for `call_expr` nodes
    mySys.argv # OK

def optparse_test():
    OptionParser() # Noncompliant
    optparse.OptionParser() # Noncompliant