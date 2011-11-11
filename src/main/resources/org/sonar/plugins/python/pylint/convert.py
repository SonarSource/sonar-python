# Sonar Python Plugin
# Copyright (C) 2011 Waleri Enns
# Author(s) : Waleri Enns
# waleri.enns@gmail.com

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 3 of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public
# License along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02


#
# Reads pylint --list-msgs from the stdin and writes XML to the stdout.
# The latter is compatible with sonar rules.xml schema.
#

import sys
import re

RULEID_PATTERN = ":([A-Z][0-9]{4}):"

def parseNextRule(lines):
    ruleid = grabId(lines)
    ruledescr = grabDescr(lines)
    if ruleid:
        if not ruledescr:
            raise Exception("Invalid input")
        return Rule(ruleid, ruledescr)
    return None

def grabId(lines):
    if not lines:
        return ""
    currline = lines.pop()
    ruleid = ""
    match = re.match(RULEID_PATTERN, currline)
    if match:
        ruleid = match.groups()[0]
    return ruleid

def grabDescr(lines):
    def partOfDescr(line):
        return line[:2] == "  "
    
    descr = ""
    while lines:
        currline = lines.pop()
        if partOfDescr(currline):
            descr += currline.strip()
        else:
            lines.append(currline)
            break
    return descr


def header():
    return ('<?xml version="1.0" encoding="ASCII"?>\n'
            '<!--\n'
            'Automatically generated from the output of "pylint --list-msgs"\n'
            '-->\n'
            '<rules>\n')


def footer():
    return "</rules>\n"


class Rule:
    def __init__(self, ruleid, ruledescr):
        self.ruleid = ruleid
        self.ruledescr = ruledescr

    def toxml(self):
        rid = self.ruleid
        return ("<rule>\n"
                "<key>%s</key>\n"
                "<name>%s</name>\n"
                "<configKey>%s</configKey>\n"
                "<description>\n"
                "<![CDATA[%s]]>\n"
                "</description>\n"
                "</rule>\n"
                % (self.ruleid, self.ruleid, self.ruleid, self.ruledescr))


lines = sys.stdin.readlines()
lines.reverse()

# parse line oriented input format
rules = []
while(lines):
    rule = parseNextRule(lines)
    if rule:
        rules.append(rule)

# generate rules-file as expected by sonar
outstream = sys.stdout
outstream.write(header())
for rule in rules:
    outstream.write(rule.toxml())
outstream.write(footer())
