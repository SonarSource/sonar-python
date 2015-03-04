import collections
import sys

Rule = collections.namedtuple("Rule", ['code', 'message'])
    
def parse_rule(line):
    (code, message) = line.split(' ', 1)
    assert code
    assert message
    return Rule(code, message.strip())

def convert_rule_to_xml(rule):
    xml = '\n'.join([
        "  <rule>",
        "      <key>{code}</key>",
        "      <name><![CDATA[{message}]]></name>",
        "      <configKey>{code}</configKey>",
        "      <description>",
        "          <![CDATA[{message}]]>",
        "      </description>",
        "  </rule>"])
    return xml.format(code=rule.code, message=rule.message)


xml = ('<?xml version="1.0" encoding="ASCII"?>\n'
       '<rules>')
for line in sys.stdin.readlines():
    rule = parse_rule(line)
    xml += "\n%s" % convert_rule_to_xml(rule)
xml += '\n</rules>'

sys.stdout.write(xml)