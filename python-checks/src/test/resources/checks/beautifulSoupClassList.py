from bs4 import BeautifulSoup

html = "<div class='A B'>test</div>"
soup = BeautifulSoup(html, 'html.parser')

# --- List-returning methods → "Use select()" ---

result = soup.find_all('div', class_=['A', 'B'])  # Noncompliant {{Use "select()" with a chained CSS class selector instead.}}
#                             ^^^^^^^^^^^^^^^^^
result = soup.findAll('div', class_=['A', 'B'])   # Noncompliant {{Use "select()" with a chained CSS class selector instead.}}
result = soup.find_all(class_=['A', 'B', 'C'])   # Noncompliant {{Use "select()" with a chained CSS class selector instead.}}
result = soup.find_all('div', class_=['A', 'B', 'C'])  # Noncompliant {{Use "select()" with a chained CSS class selector instead.}}
#                             ^^^^^^^^^^^^^^^^^^^^^^
result = soup.findChildren('div', class_=['A', 'B'])  # Noncompliant {{Use "select()" with a chained CSS class selector instead.}}

result = soup.find_parents('div', class_=['A', 'B'])  # Noncompliant {{Use "select()" with a chained CSS class selector instead.}}
result = soup.findParents('div', class_=['A', 'B'])  # Noncompliant {{Use "select()" with a chained CSS class selector instead.}}

result = soup.find_all_next('div', class_=['A', 'B'])  # Noncompliant {{Use "select()" with a chained CSS class selector instead.}}
result = soup.findAllNext('div', class_=['A', 'B'])  # Noncompliant {{Use "select()" with a chained CSS class selector instead.}}

result = soup.find_next_siblings('div', class_=['A', 'B'])  # Noncompliant {{Use "select()" with a chained CSS class selector instead.}}
result = soup.findNextSiblings('div', class_=['A', 'B'])  # Noncompliant {{Use "select()" with a chained CSS class selector instead.}}

result = soup.find_all_previous('div', class_=['A', 'B'])  # Noncompliant {{Use "select()" with a chained CSS class selector instead.}}
result = soup.findAllPrevious('div', class_=['A', 'B'])  # Noncompliant {{Use "select()" with a chained CSS class selector instead.}}

result = soup.find_previous_siblings('div', class_=['A', 'B'])  # Noncompliant {{Use "select()" with a chained CSS class selector instead.}}
result = soup.findPreviousSiblings('div', class_=['A', 'B'])  # Noncompliant {{Use "select()" with a chained CSS class selector instead.}}

# --- Single-result methods → "Use select_one()" ---

result = soup.find('div', class_=['A', 'B'])  # Noncompliant {{Use "select_one()" with a chained CSS class selector instead.}}
result = soup.findChild('div', class_=['A', 'B'])  # Noncompliant {{Use "select_one()" with a chained CSS class selector instead.}}

result = soup.find_parent('div', class_=['A', 'B'])  # Noncompliant {{Use "select_one()" with a chained CSS class selector instead.}}
result = soup.findParent('div', class_=['A', 'B'])  # Noncompliant {{Use "select_one()" with a chained CSS class selector instead.}}

result = soup.find_next('div', class_=['A', 'B'])  # Noncompliant {{Use "select_one()" with a chained CSS class selector instead.}}
result = soup.findNext('div', class_=['A', 'B'])  # Noncompliant {{Use "select_one()" with a chained CSS class selector instead.}}

result = soup.find_next_sibling('div', class_=['A', 'B'])  # Noncompliant {{Use "select_one()" with a chained CSS class selector instead.}}
result = soup.findNextSibling('div', class_=['A', 'B'])  # Noncompliant {{Use "select_one()" with a chained CSS class selector instead.}}

result = soup.find_previous('div', class_=['A', 'B'])  # Noncompliant {{Use "select_one()" with a chained CSS class selector instead.}}
result = soup.findPrevious('div', class_=['A', 'B'])  # Noncompliant {{Use "select_one()" with a chained CSS class selector instead.}}

result = soup.find_previous_sibling('div', class_=['A', 'B'])  # Noncompliant {{Use "select_one()" with a chained CSS class selector instead.}}
result = soup.findPreviousSibling('div', class_=['A', 'B'])  # Noncompliant {{Use "select_one()" with a chained CSS class selector instead.}}

# Compliant: use CSS selector instead
result = soup.select('div.A.B')
result = soup.select_one('div.A.B')

# Compliant: single class string
result = soup.find_all('div', class_='A')
result = soup.find('div', class_='A')
result = soup.find_parent('div', class_='A')
result = soup.find_next('div', class_='A')
result = soup.find_next_sibling('div', class_='A')
result = soup.find_previous('div', class_='A')
result = soup.find_previous_sibling('div', class_='A')

# Compliant: single-element list (no ambiguity)
result = soup.find_all('div', class_=['A'])
result = soup.find('div', class_=['A'])

# Compliant: no class_ argument
result = soup.find_all('div')
result = soup.find('div')
result = soup.find_all('div', other=['A', 'B'])

def not_beautifulsoup(some_obj):
    # Compliant: not a bs4 object
    result = some_obj.find_all('div', class_=['A', 'B'])
    result = some_obj.findAll('div', class_=['A', 'B'])
    result = some_obj.find('div', class_=['A', 'B'])
    result = some_obj.findChild('div', class_=['A', 'B'])
    result = some_obj.find_parent('div', class_=['A', 'B'])
    result = some_obj.findParent('div', class_=['A', 'B'])
