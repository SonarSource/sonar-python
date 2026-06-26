from bs4 import BeautifulSoup
from bs4.element import Tag

html = "<html><body><p>Hello</p><a href='http://example.com'>link</a></body></html>"


# --- Noncompliant: inline chaining on all 7 search methods ---

def noncompliant_find_attr():
    soup = BeautifulSoup(html, "html.parser")
    text = soup.find("p").text  # Noncompliant {{Check if this element exists before accessing it with `.text`.}}
        #       ^^^^

def noncompliant_find_subscript():
    soup = BeautifulSoup(html, "html.parser")
    cls = soup.find("a")["class"]  # Noncompliant {{Check if this element exists before accessing it with `[class]`.}}
    a = "access"
    cls = soup.find("a")[a]  # Noncompliant {{Check if this element exists before accessing it with `[]`.}}


def noncompliant_select_one_attr():
    soup = BeautifulSoup(html, "html.parser")
    text = soup.select_one("p").text  # Noncompliant


def noncompliant_find_next_attr():
    soup = BeautifulSoup(html, "html.parser")
    text = soup.find_next("p").text  # Noncompliant


def noncompliant_find_previous_attr():
    soup = BeautifulSoup(html, "html.parser")
    text = soup.find_previous("p").text  # Noncompliant


def noncompliant_find_next_sibling_attr():
    soup = BeautifulSoup(html, "html.parser")
    text = soup.find_next_sibling("p").text  # Noncompliant


def noncompliant_find_previous_sibling_attr():
    soup = BeautifulSoup(html, "html.parser")
    text = soup.find_previous_sibling("p").text  # Noncompliant


def noncompliant_find_parent_attr():
    soup = BeautifulSoup(html, "html.parser")
    text = soup.find_parent("div").text  # Noncompliant


# --- Noncompliant: method call on search result ---

def noncompliant_method_call_inline():
    soup = BeautifulSoup(html, "html.parser")
    text = soup.find("p").get_text()  # Noncompliant


def noncompliant_method_call_variable():
    soup = BeautifulSoup(html, "html.parser")
    element = soup.find("p")
    text = element.get_text()  # Noncompliant


# --- Noncompliant: chained BS4 search calls ---

def noncompliant_chained_find():
    soup = BeautifulSoup(html, "html.parser")
    text = soup.find("div").find("p").text  # Noncompliant {{Check if this element exists before accessing it with `.find`.}}
        #       ^^^^


def noncompliant_chained_find_no_attr():
    soup = BeautifulSoup(html, "html.parser")
    inner = soup.find("div").find("p")  # Noncompliant {{Check if this element exists before accessing it with `.find`.}}
        #        ^^^^


def noncompliant_cond_without_return_stmt(cond):
    soup = BeautifulSoup(html, "html.parser")
    element = soup.find("p")
    if not element:
       pass 
    text = element.text  # Noncompliant
#
# --- Noncompliant: "is not <non-None>" is not a None-guard ---

def noncompliant_is_not_non_none(other):
    soup = BeautifulSoup(html, "html.parser")
    element = soup.find("p")
    if element is not other:
        text = element.text  # Noncompliant


# --- Noncompliant: variable assignment then unguarded access ---

def noncompliant_variable_attr():
    soup = BeautifulSoup(html, "html.parser")
    element = soup.find("p")
    text = element.text  # Noncompliant
        #          ^^^^

def noncompliant_variable_subscript():
    soup = BeautifulSoup(html, "html.parser")
    element = soup.find("a")
    cls = element["class"]  # Noncompliant
    #     ^^^^^^^^^^^^^^^^

def noncompliant_variable_select_one():
    soup = BeautifulSoup(html, "html.parser")
    element = soup.select_one("p")
    text = element.text  # Noncompliant


# --- Compliant: guarded with "if element is not None:" ---

def compliant_is_not_none():
    soup = BeautifulSoup(html, "html.parser")
    element = soup.find("p")
    if element is not None:
        text = element.text  # Compliant
        cls = element["class"]  # Compliant


# --- Compliant: guarded with "if element:" ---

def compliant_truthiness():
    soup = BeautifulSoup(html, "html.parser")
    element = soup.find("p")
    if element:
        text = element.text  # Compliant


# --- Compliant: nested if inside a None-guard ---

def compliant_nested_if():
    soup = BeautifulSoup(html, "html.parser")
    element = soup.find("p")
    if element is not None:
        if True:
            text = element.text  # Compliant


# --- Compliant: compound "and" condition (guard on left or right) ---

def compliant_and_condition_guard_left():
    soup = BeautifulSoup(html, "html.parser")
    element = soup.find("p")
    if element is not None and True:
        text = element.text  # Compliant


def compliant_and_condition_guard_right():
    soup = BeautifulSoup(html, "html.parser")
    element = soup.find("p")
    if True and element is not None:
        text = element.text  # Compliant


# --- Compliant: guarded with "if element != None:" ---

def compliant_not_equal_none():
    soup = BeautifulSoup(html, "html.parser")
    element = soup.find("p")
    if element != None:
        text = element.text  # Compliant


# --- Compliant: subscription as assignment target (write, not read) ---

def compliant_subscription_assignment_target():
    soup = BeautifulSoup(html, "html.parser")
    element = soup.find("p")
    if element:
        element["class"] = "active"  # Compliant


# --- Compliant: early-return guard with "if element == None:" ---

def compliant_early_return_equal_none():
    soup = BeautifulSoup(html, "html.parser")
    element = soup.find("p")
    if element == None:
        return
    text = element.text  # Compliant


def compliant_early_return_none_equal():
    soup = BeautifulSoup(html, "html.parser")
    element = soup.find("p")
    if None == element:
        return
    text = element.text  # Compliant


def compliant_early_return_none_is():
    soup = BeautifulSoup(html, "html.parser")
    element = soup.find("p")
    if None is element:
        return
    text = element.text  # Compliant


# --- Compliant: early-return guard ---

def compliant_early_return_is_none():
    soup = BeautifulSoup(html, "html.parser")
    element = soup.find("p")
    if element is None:
        return
    text = element.text  # Compliant


def compliant_early_return_not():
    soup = BeautifulSoup(html, "html.parser")
    element = soup.find("p")
    if not element:
        return
    text = element.text  # Compliant


# --- Compliant: "None != element" guard ---

def compliant_none_not_equal():
    soup = BeautifulSoup(html, "html.parser")
    element = soup.find("p")
    if None != element:
        text = element.text  # Compliant


# --- Noncompliant: inverted guard (early-return on is-not-None leaves only None path) ---

def noncompliant_early_return_is_not_none():
    soup = BeautifulSoup(html, "html.parser")
    element = soup.find("p")
    if element is not None:
        return
    text = element.text  # Noncompliant


# --- Noncompliant: assert element is None (asserts None, then accesses) ---

def noncompliant_assert_is_none():
    soup = BeautifulSoup(html, "html.parser")
    element = soup.find("p")
    assert element is None
    text = element.text  # Noncompliant


# --- Noncompliant: dynamic subscript ([] fallback description) ---

def noncompliant_dynamic_subscript():
    soup = BeautifulSoup(html, "html.parser")
    key = "class"
    cls = soup.find("a")[key]  # Noncompliant {{Check if this element exists before accessing it with `[]`.}}


# --- Compliant: unresolved name (valuesAtLocation returns empty) ---

def compliant_unresolved_name(element):
    # element is a parameter — valuesAtLocation returns empty, no issue raised
    text = element.text  # Compliant


# --- Compliant: assert guard ---

def compliant_assert_truthy():
    soup = BeautifulSoup(html, "html.parser")
    element = soup.find("p")
    assert element
    text = element.text  # Compliant


def compliant_assert_is_not_none():
    soup = BeautifulSoup(html, "html.parser")
    element = soup.find("p")
    assert element is not None
    text = element.text  # Compliant


# --- Compliant: inline chaining inside try/except catching AttributeError ---

def compliant_try_except_attribute_error():
    soup = BeautifulSoup(html, "html.parser")
    try:
        text = soup.find("p").text  # Compliant
    except AttributeError:
        pass


def noncompliant_try_except_exception():
    soup = BeautifulSoup(html, "html.parser")
    try:
        text = soup.find("p").text  # Noncompliant
    except Exception:
        pass


def noncompliant_try_except_bare():
    soup = BeautifulSoup(html, "html.parser")
    try:
        text = soup.find("p").text  # Noncompliant
    except:
        pass


def compliant_try_except_tuple():
    soup = BeautifulSoup(html, "html.parser")
    try:
        text = soup.find("p").text  # Compliant
    except (AttributeError, TypeError):
        pass


# --- Noncompliant: inline chaining inside try/except catching unrelated exception ---

def noncompliant_try_except_unrelated():
    soup = BeautifulSoup(html, "html.parser")
    try:
        text = soup.find("p").text  # Noncompliant
    except ValueError:
        pass


# --- Compliant: non-BS4 object ---

def compliant_non_bs4():
    d = {"key": "value"}
    val = d["key"]  # Compliant
    text = d.get("key")  # Compliant

