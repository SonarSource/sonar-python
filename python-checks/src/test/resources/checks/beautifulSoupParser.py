from bs4 import BeautifulSoup

html = "<html><body><p>test</p></body></html>"

soup = BeautifulSoup(html)  # Noncompliant {{Specify the parser to use for "BeautifulSoup".}}
#      ^^^^^^^^^^^^^
soup = BeautifulSoup()  # Noncompliant
#      ^^^^^^^^^^^^^
soup = BeautifulSoup(html, None)
soup = BeautifulSoup(html, features=None)

# Compliant: features specified as second positional arg
soup = BeautifulSoup(html, 'html.parser')
soup = BeautifulSoup(html, 'lxml')
soup = BeautifulSoup(html, 'html5lib')

# Compliant: features specified as keyword argument
soup = BeautifulSoup(html, features='html.parser')
soup = BeautifulSoup(features='html.parser', markup=html)

# Compliant: builder specified (alternative way to pin the parser)
from bs4.builder import HTMLParserTreeBuilder
soup = BeautifulSoup(html, builder=None)
soup = BeautifulSoup(html, builder=HTMLParserTreeBuilder())
soup = BeautifulSoup(html, None, HTMLParserTreeBuilder())  # builder as 3rd positional

# Compliant: not a bs4 BeautifulSoup
class BeautifulSoup:
    def __init__(self, markup): pass

soup = BeautifulSoup(html)  # Not bs4 - no issue
