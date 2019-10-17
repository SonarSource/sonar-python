#!/usr/bin/python

# Various stuff from the sys module.
import sys, os, glob

print "sys.argv:", sys.argv
print "os.environ:"
for v in os.environ.keys():
    print "    %-15s => %s" % (v, os.environ[v])
print "sys.platform:", sys.platform
print "os.getcwd:", os.getcwd()
print "os.listdir('.'):", os.listdir('.')
print "glob.glob('*.py'):", glob.glob('*.py')
print "sys:", dir(sys)
print "os:", dir(os)
print

os.system('ls -l')
