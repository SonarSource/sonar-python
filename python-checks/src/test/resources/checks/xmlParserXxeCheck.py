from lxml import etree

def case1():
    parser = etree.XMLParser() # Noncompliant {{Disable access to external entities in XML parsing.}}
    #        ^^^^^^^^^^^^^^^^^
    tree1 = etree.parse('ressources/xxe.xml', parser)
    #       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^<

def case2():
    parser = etree.XMLParser(resolve_entities=True) # Noncompliant
    #        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    tree1 = etree.parse('ressources/xxe.xml', parser)
    #       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^<

def case3():
    tree1 = etree.parse('ressources/xxe.xml', etree.XMLParser(resolve_entities=True)) # Noncompliant
    #      >^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def case4():
    parser = etree.XMLParser(resolve_entities=False) # compliant
    tree1 = etree.parse('ressources/xxe.xml', parser)  # compliant

def case5():
    parser = etree.XMLParser(no_network=False) # Noncompliant
    #        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    tree1 = etree.parse('ressources/xxe.xml', parser)
    #       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^<

def case6():
    parser = etree.XMLParser(resolve_entities=False) # Compliant by default no_network=True
    tree1 = etree.parse('ressources/xxe.xml', parser)  # Compliant

def case7():
    parser = etree.XMLParser(resolve_entities=False, no_network=True) # compliant
    tree1 = etree.parse('ressources/xxe.xml', parser)  # Compliant

def case8():
    ac = etree.XSLTAccessControl(read_network=True, write_network=False)  # Noncompliant
    #    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    transform = etree.XSLT(rootxsl, access_control=ac)
    #           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^<

def case9():
    ac = etree.XSLTAccessControl(read_network=False, write_network=False)  # Compliant
    transform = etree.XSLT(rootxsl, access_control=ac) # Compliant

def case10():
    ac = etree.XSLTAccessControl.DENY_ALL # Compliant
    transform = etree.XSLT(rootxsl, access_control=ac) # Compliant

def case11():
    ac = etree.XSLTAccessControl()  # Noncompliant
    #    ^^^^^^^^^^^^^^^^^^^^^^^^^
    transform = etree.XSLT(rootxsl, access_control=ac)
    #           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^<

def case12():
    transform = etree.XSLT(rootxsl) # Noncompliant
    #           ^^^^^^^^^^^^^^^^^^^

def case13():
    transform = etree.XSLT(rootxsl, access_control=etree.XSLTAccessControl()) # Noncompliant
    #           >^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def case14():
    transform = etree.XSLT(rootxsl, access_control=ac)

def case15():
    ac = foo()
    transform = etree.XSLT(rootxsl, access_control=ac)

def case16():
    tree1 = etree.parse('ressources/xxe.xml', parser)
    tree1 = etree.parse('ressources/xxe.xml')
    tree1 = etree.parse('ressources/xxe.xml', a.b.c)

def case17():
    parser = foo()
    tree1 = etree.parse('ressources/xxe.xml', parser)

import xml.sax

def case18():
    parser = xml.sax.make_parser()
    #       >^^^^^^^^^^^^^^^^^^^^^
    parser.setFeature(feature_external_ges, True) # Noncompliant
   #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def case19():
    parser.setFeature()
    parser.setFeature(feature_external_ges)
    parser.setFeature(feature_external_ges, True)
    parser.setFeature(feature_external_ges, False)
    parser.setFeature(feature_external_ges123, False)
    parser.setFeature(foo(), False)

def case20():
    parser.setFeature(feature_external_ges, True)

def case21():
    parser = foo()
    parser.setFeature(feature_external_ges, True)
    prs = 42
    prs.setFeature(feature_external_ges, True)
