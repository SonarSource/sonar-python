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
    argv # Noncompliant
    mySys.argv # OK

def optparse_test():
    OptionParser() # Noncompliant
    optparse.OptionParser() # Noncompliant



def assign_to_argv(original_argv):
    sys.argv = original_argv # OK

def calling_list_methods():
    sys.argv.index('-d') # Noncompliant
    sys.argv.remove('test') # Noncompliant
    sys.argv.append('test') # Noncompliant
    sys.argv.extend(['config_fc'] + []) # Noncompliant
