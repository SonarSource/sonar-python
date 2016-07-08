""" Docstring
    for the module """

x = "some string"

# This is a comment

def foo():
    pass  # other comment

# the line below has 4 keywords: if, or, or, continue
if t0 == t1 or t1 == t2 or t2 == t0: continue

''' bla bla '''

""" bla bla """

y = 'some string'

y = """ some string
        that extends
        on several
        lines
    """

values=["""long...
    ...string 1""", 3.14, "short string 2"]

34

-35

20000000000000L

1000l

89e4

y = 45.4 + 67e8 - 78.562E-09

4.55j

-4.55j

3J

23.3e-7J

def foo():
    "Docstring for the function"
    pass

def func():
    '''
    Multi-line docstring
    for the function
    '''
    "Additional docstring, currently not highlighted as a docstring"
    print(1)
    """ Not a docstring """
    pass

class ClassA:
    """ Docstring for the class """

    attr1 = 1
    """ Attribute docstring, currently not highlighted as a docstring """

    def meth1(self):
        ''' Multi-line docstring
            for 
            the method'''
        pass

    def meth2(self):
        print(1)
        """ Not a docstring """
        while (condition):
            """ Not a docstring """
            print(1)
        pass

x = 1
""" Not a docstring """

if (condition):
    """ Not a docstring """
    pass
    
def foo():
    foo("Not a docstring")

def foo(): "Not a docstring"
