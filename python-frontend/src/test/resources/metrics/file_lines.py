# comment
"""A module-level docstring.

Following PEP-257.
"""
if 2 > 1:
	
	print "b"     # end of line comment

# strings
u"lala"
U"lala"
r"lala"
R"lala"
print             # NOSONAR
a = [1,
     # inline comment
     2,           # end of line comment
     ]
#
def foo1(x,y,z,): # end of line comment
    """
    3 lines of docstring
    """
    pass


def foo2(x,y,z,):
    """
    3 lines of docstring
    """ # NOSONAR
    pass

"some string literal" # NOSONAR

def foo3():
    """
    some tuple
    """, """
    of multiline strings
    """ # NOSONAR

def foo4():
    (multine,
    expression) # NOSONAR
