from bs4 import BeautifulSoup, Tag

html = "<html><body><a href='#'>Click here</a></body></html>"
soup = BeautifulSoup(html, 'html.parser')
# Construct a Tag directly so the type inference yields a definite 'bs4.element.Tag' type
tag = Tag(name='a')


# Deprecated method calls on BeautifulSoup
def deprecated_methods_on_soup():
    soup.findAll('a')  # Noncompliant {{Replace the deprecated 'findAll()' method with 'find_all()'.}}
    soup.findChild('a')  # Noncompliant {{Replace the deprecated 'findChild()' method with 'find()'.}}
    soup.findChildren('a')  # Noncompliant {{Replace the deprecated 'findChildren()' method with 'find_all()'.}}
    soup.findNext('a')  # Noncompliant {{Replace the deprecated 'findNext()' method with 'find_next()'.}}
    soup.findAllNext('a')  # Noncompliant {{Replace the deprecated 'findAllNext()' method with 'find_all_next()'.}}
    soup.findPrevious('a')  # Noncompliant {{Replace the deprecated 'findPrevious()' method with 'find_previous()'.}}
    soup.findAllPrevious('a')  # Noncompliant {{Replace the deprecated 'findAllPrevious()' method with 'find_all_previous()'.}}
    soup.findNextSibling('a')  # Noncompliant {{Replace the deprecated 'findNextSibling()' method with 'find_next_sibling()'.}}
    soup.findNextSiblings('a')  # Noncompliant {{Replace the deprecated 'findNextSiblings()' method with 'find_next_siblings()'.}}
    soup.findPreviousSibling('a')  # Noncompliant {{Replace the deprecated 'findPreviousSibling()' method with 'find_previous_sibling()'.}}
    soup.findPreviousSiblings('a')  # Noncompliant {{Replace the deprecated 'findPreviousSiblings()' method with 'find_previous_siblings()'.}}
    soup.findParent('a')  # Noncompliant {{Replace the deprecated 'findParent()' method with 'find_parent()'.}}
    soup.findParents('a')  # Noncompliant {{Replace the deprecated 'findParents()' method with 'find_parents()'.}}
    soup.replaceWith('other')  # Noncompliant {{Replace the deprecated 'replaceWith()' method with 'replace_with()'.}}
    soup.getText()  # Noncompliant {{Replace the deprecated 'getText()' method with 'get_text()'.}}


# Deprecated method calls on Tag (find() returns a Tag, which is also a PageElement)
def deprecated_methods_on_tag():
    tag.findAll('b')  # Noncompliant {{Replace the deprecated 'findAll()' method with 'find_all()'.}}
    tag.findChild('b')  # Noncompliant {{Replace the deprecated 'findChild()' method with 'find()'.}}
    tag.findNext('b')  # Noncompliant {{Replace the deprecated 'findNext()' method with 'find_next()'.}}
    tag.getText()  # Noncompliant {{Replace the deprecated 'getText()' method with 'get_text()'.}}


# Deprecated attribute access
def deprecated_attributes():
    _ = soup.nextSibling  # Noncompliant {{Replace the deprecated 'nextSibling' attribute with 'next_sibling'.}}
    _ = soup.previousSibling  # Noncompliant {{Replace the deprecated 'previousSibling' attribute with 'previous_sibling'.}}
    _ = tag.nextSibling  # Noncompliant {{Replace the deprecated 'nextSibling' attribute with 'next_sibling'.}}
    _ = tag.previousSibling  # Noncompliant {{Replace the deprecated 'previousSibling' attribute with 'previous_sibling'.}}


# Deprecated text= keyword argument (on modern method names)
def deprecated_text_kwarg():
    soup.find('a', text='Click here')  # Noncompliant {{Replace the deprecated 'text' keyword argument with 'string'.}}
    soup.find_all('a', text='Click here')  # Noncompliant {{Replace the deprecated 'text' keyword argument with 'string'.}}
    soup.find_next('a', text='x')  # Noncompliant {{Replace the deprecated 'text' keyword argument with 'string'.}}
    soup.find_all_next('a', text='x')  # Noncompliant {{Replace the deprecated 'text' keyword argument with 'string'.}}
    soup.find_previous('a', text='x')  # Noncompliant {{Replace the deprecated 'text' keyword argument with 'string'.}}
    soup.find_all_previous('a', text='x')  # Noncompliant {{Replace the deprecated 'text' keyword argument with 'string'.}}
    soup.find_next_sibling('a', text='x')  # Noncompliant {{Replace the deprecated 'text' keyword argument with 'string'.}}
    soup.find_next_siblings('a', text='x')  # Noncompliant {{Replace the deprecated 'text' keyword argument with 'string'.}}
    soup.find_previous_sibling('a', text='x')  # Noncompliant {{Replace the deprecated 'text' keyword argument with 'string'.}}
    soup.find_previous_siblings('a', text='x')  # Noncompliant {{Replace the deprecated 'text' keyword argument with 'string'.}}
    soup.find_parent('a', text='x')  # Compliant - find_parent does not accept a string= argument
    soup.find_parents('a', text='x')  # Compliant - find_parents does not accept a string= argument


# Deprecated method + deprecated text= argument = two issues on same call
def deprecated_method_and_text_kwarg():
    soup.findAll('a', text='Click here')  # Noncompliant {{Replace the deprecated 'findAll()' method with 'find_all()'.}}
    # Noncompliant@-1 {{Replace the deprecated 'text' keyword argument with 'string'.}}


# Compliant: modern method names
def compliant_methods():
    soup.find('a')
    soup.find_all('a')
    soup.find_next('a')
    soup.find_all_next('a')
    soup.find_previous('a')
    soup.find_all_previous('a')
    soup.find_next_sibling('a')
    soup.find_next_siblings('a')
    soup.find_previous_sibling('a')
    soup.find_previous_siblings('a')
    soup.find_parent('a')
    soup.find_parents('a')
    soup.replace_with('other')
    soup.get_text()


# Compliant: modern attribute names
def compliant_attributes():
    _ = soup.next_sibling
    _ = soup.previous_sibling
    _ = tag.next_sibling
    _ = tag.previous_sibling


# Compliant: modern string= keyword
def compliant_string_kwarg():
    soup.find('a', string='Click here')
    soup.find_all('a', string='Click here')


# Compliant: not a bs4 object
def compliant_non_bs4():
    class FakeElement:
        def findAll(self):
            pass
        nextSibling = None

    obj = FakeElement()
    obj.findAll()  # Compliant - not a bs4 type
    _ = obj.nextSibling  # Compliant - not a bs4 type
