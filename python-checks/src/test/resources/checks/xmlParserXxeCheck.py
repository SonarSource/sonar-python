from lxml import etree

def xml_parser_simple():
    parser = etree.XMLParser() # Noncompliant {{Disable access to external entities in XML parsing.}}
    #        ^^^^^^^^^^^^^^^^^
    tree1 = etree.parse('ressources/xxe.xml', parser)
    #       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^<  {{This function loads the XML code and triggers the vulnerability.}}

def xml_parser_simple_resolve_entities():
    parser = etree.XMLParser(resolve_entities=True) # Noncompliant
    #        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    tree1 = etree.parse('ressources/xxe.xml', parser)
    #       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^<

def xml_parser_simple_resolve_entities_inline():
    tree1 = etree.parse('ressources/xxe.xml', etree.XMLParser(resolve_entities=True)) # Noncompliant
    #      >^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ {{This function loads the XML code and triggers the vulnerability.}}

def xml_parser_resolve_entities_off():
    parser = etree.XMLParser(resolve_entities=False) # Compliant (default is no_network=True)
    tree1 = etree.parse('ressources/xxe.xml', parser)  # Compliant

def xml_parser_no_network():
    parser = etree.XMLParser(no_network=False) # Noncompliant
    #        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    tree1 = etree.parse('ressources/xxe.xml', parser)
    #       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^<

def xml_parser_resolve_entities_off_no_network():
    parser = etree.XMLParser(resolve_entities=False, no_network=True) # compliant
    tree1 = etree.parse('ressources/xxe.xml', parser)  # Compliant

def xslt_ac_write_network_off():
    ac = etree.XSLTAccessControl(read_network=True, write_network=False)  # Noncompliant
    #    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    transform = etree.XSLT(rootxsl, access_control=ac)
    #           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^<

def xslt_ac_read_write_network_off():
    ac = etree.XSLTAccessControl(read_network=False, write_network=False)  # Compliant
    transform = etree.XSLT(rootxsl, access_control=ac) # Compliant

def xslt_ac_deny_all():
    ac = etree.XSLTAccessControl.DENY_ALL # Compliant
    transform = etree.XSLT(rootxsl, access_control=ac) # Compliant

def xslt_ac_default():
    ac = etree.XSLTAccessControl()  # Noncompliant
    #    ^^^^^^^^^^^^^^^^^^^^^^^^^
    transform = etree.XSLT(rootxsl, access_control=ac)
    #           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^<

def xslt_no_ac():
    transform = etree.XSLT(rootxsl) # Noncompliant
    #           ^^^^^^^^^^^^^^^^^^^

def xslt_inline_ac():
    transform = etree.XSLT(rootxsl, access_control=etree.XSLTAccessControl()) # Noncompliant
    #           >^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def xslt_no_such_ac():
    transform = etree.XSLT(rootxsl, access_control=no_such_ac)

def xslt_ac_unknown_source():
    ac = no_such_function()
    transform = etree.XSLT(rootxsl, access_control=ac)

def parse_edge_cases():
    tree1 = etree.parse('ressources/xxe.xml', no_such_parser)
    tree1 = etree.parse('ressources/xxe.xml')
    tree1 = etree.parse('ressources/xxe.xml', a.b.c)

def parse_unknown_parser_source():
    parser = no_such_function()
    tree1 = etree.parse('ressources/xxe.xml', parser)

def parse_unpacking_args():
    parser = etree.XMLParser()
    args = ['ressources/xxe.xml', parser]
    etree.parse(*args) # FN

import xml.sax

from xml.sax.handler import feature_external_ges
import xml.sax.handler as sax_handler

def set_feature():
    parser = xml.sax.make_parser()
    #       >^^^^^^^^^^^^^^^^^^^^^
    parser.setFeature(xml.sax.handler.feature_external_ges, True) # Noncompliant
   #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    parser.setFeature(feature_external_ges, True) # Noncompliant
    parser.setFeature(sax_handler.feature_external_ges, True) # Noncompliant
    parser.setFeature(sax_handler.feature_external_ges, False) # Ok
    parser.setFeature(no_such_module.feature_external_ges, True) # Ok

    foo = ''
    parser.setFeature(foo, True) # Ok

def set_feature_unknown_parser():
    no_such_parser.setFeature()
    no_such_parser.setFeature(xml.sax.handler.feature_external_ges)
    no_such_parser.setFeature(xml.sax.handler.feature_external_ges, True)
    no_such_parser.setFeature(xml.sax.handler.feature_external_ges, False)
    no_such_parser.setFeature(xml.sax.handler.feature_external_ges, False)
    no_such_parser.setFeature(foo(), False)

def set_feature_edge_cases():
    parser = no_such_function()
    parser.setFeature(feature_external_ges, True)
    args = ['foo', 'bar']
    parser.setFeature(*args, 'baz')
    prs = 42
    prs.setFeature(feature_external_ges, True)
