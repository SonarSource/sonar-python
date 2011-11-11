#!/usr/bin/env python

import os
import sys
from glob import glob
from optparse import OptionParser

import cc


COMMANDS = ['all', 'complexity', ]
USAGE = 'usage: pygenie command [directories|files|packages]'


class CommandParser(object):

    def __init__ (self, optparser, commands):
        self.commands = commands or []
        self.optparser = optparser

    def parse_args(self, args=None, values=None):
        args = args or sys.argv[1:]
        if len(args) < 1:
            self.optparser.error('please provide a valid command')

        command = args[0]
        if command not in self.commands:
            self.optparser.error("'%s' is not a valid command" % command)
            
        options, values = self.optparser.parse_args(args[1:], values)
        return command, options, values


def find_module(fqn):
    join = os.path.join
    exists = os.path.exists
    partial_path = fqn.replace('.', os.path.sep)
    for p in sys.path:
        path = join(p, partial_path, '__init__.py')
        if exists(path):
            return path
        path = join(p, partial_path + '.py')
        if exists(path):
            return path
    raise Exception('invalid module')


def main():
    from optparse import OptionParser

    parser = OptionParser(usage='./cc.py command [options] *.py')
    parser.add_option('-v', '--verbose',
            dest='verbose', action='store_true', default=False,
            help='print detailed statistics to stdout')
    parser = CommandParser(parser, COMMANDS)
    command, options, args = parser.parse_args()

    items = set()
    for arg in args: 
        if os.path.isdir(arg):
            for f in glob(os.path.join(arg, '*.py')):
                if os.path.isfile(f):
                    items.add(os.path.abspath(f))
        elif os.path.isfile(arg):
            items.add(os.path.abspath(arg))
        else:
            # this should be a package'
            items.add(find_module(arg))

    for item in items:
        code = open(item).read()
        if command in ('all', 'complexity'):
            stats = cc.measure_complexity(code, item)
            pp = cc.PrettyPrinter(sys.stdout, verbose=options.verbose)
            pp.pprint(item, stats)


if __name__ == '__main__':
    main()

