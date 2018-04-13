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
# The latter is compatible with sonar rules.xml-schema.
#

import sys
import re

RULEID_PATTERN = ":[a-z _-]+ \(([A-Z][0-9]{4})\): ?\*?(.*?)\*?$"

def parseNextRule(lines):
    ruleid, rulename = grabIdAndName(lines)
    ruledescr = grabDescr(lines)
    if ruleid:
        if not ruledescr:
            raise Exception("Invalid input")
        return Rule(ruleid, rulename, ruledescr)
    return None

def grabIdAndName(lines):
    if not lines:
        return ""
    currline = lines.pop()
    ruleid = rulename = ""
    match = re.match(RULEID_PATTERN, currline)
    if match:
        ruleid, rulename = match.groups()
    return ruleid, rulename

def grabDescr(lines):
    def partOfDescr(line):
        return line[:2] == "  "

    descr = ""
    while lines:
        currline = lines.pop()
        if partOfDescr(currline):
            descr += " " + currline.strip()
        else:
            lines.append(currline)
            break
    return descr.strip()


def header():
    return ('<?xml version="1.0" encoding="ASCII"?>\n'
            '<rules>\n')


def footer():
    return "</rules>\n"


class Rule:
    def __init__(self, ruleid, rulename, ruledescr):
        self.ruleid = ruleid
        self.rulename = rulename
        self.ruledescr = ruledescr

    def __lt__(self, other):
        return self.ruleid < other.ruleid

    def toxml(self):
        rid = self.ruleid
        return ("<rule>\n"
                "<key>%s</key>\n"
                "<name><![CDATA[%s]]></name>\n"
                "<configKey>%s</configKey>\n"
                "<description>\n"
                "<![CDATA[%s]]>\n"
                "</description>\n"
                "</rule>\n"
                % (self.ruleid, self.rulename, self.ruleid, self.ruledescr))


lines = sys.stdin.readlines()
lines.reverse()

# parse line oriented input format
rules = []
while(lines):
    rule = parseNextRule(lines)
    if rule:
        rules.append(rule)
rules.sort()

# generate rules-file as expected by sonar
outstream = sys.stdout
outstream.write(header())
for rule in rules:
    outstream.write(rule.toxml())
outstream.write(footer())
