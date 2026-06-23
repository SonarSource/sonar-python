from bs4 import BeautifulSoup
from bs4.element import Tag

soup = BeautifulSoup('<html><body></body></html>', 'html.parser')

soup.insert(0, '<div id="file_history"></div>')  # Noncompliant {{Use "new_tag()" instead of inserting raw HTML strings.}}
#    ^^^^^^
soup.append('<p>some text</p>')                  # Noncompliant {{Use "new_tag()" instead of inserting raw HTML strings.}}
tag = soup.new_tag('section')
tag.insert(0, '<span class="x"></span>')         # Noncompliant {{Use "new_tag()" instead of inserting raw HTML strings.}}
tag.append('<b>bold</b>')                        # Noncompliant {{Use "new_tag()" instead of inserting raw HTML strings.}}
tag.extend('<li>item</li>')                      # Noncompliant {{Use "new_tag()" instead of inserting raw HTML strings.}}
soup.extend(tags='<li>item</li>') # Noncompliant
# Compliant: plain text strings — idiomatic NavigableString insertion
soup.append("Hello world")   # Compliant - no markup
tag.insert(0, "some text")   # Compliant - no markup

# Variable holding an HTML string literal — value is traced back to the literal
html_fragment = '<div class="content"></div>'
tag.insert(0, html_fragment)   # Noncompliant {{Use "new_tag()" instead of inserting raw HTML strings.}}
tag.append(html_fragment)      # Noncompliant {{Use "new_tag()" instead of inserting raw HTML strings.}}

# Compliant: using new_tag() to create proper element objects
new_tag = soup.new_tag('div', id='file_history')
soup.insert(0, new_tag)   # Compliant
soup.append(new_tag)      # Compliant

# Compliant: not a string
soup.insert(0, 42)        # Compliant
soup.append(None)         # Compliant

# Compliant: extend with a list
soup.extend([new_tag])    # Compliant
soup.extend(tags=['<li>item</li>'])

soup.append("<= 3") # Compliant not the shape of a tag

# Compliant: called on a non-Tag type
my_list = [1, 2, 3]
my_list.insert(0, '<div>')   # Compliant - not a Tag
my_list.append('<p>')        # Compliant - not a Tag
