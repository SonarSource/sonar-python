import re

def non_compliant():
    input = "Bob is a Bird... Bob is a Plane... Bob is Superman!"
    changed = re.sub(r"Bob is", "It's", input)  # Noncompliant {{Replace this "re.sub()" call by a "str.replace()" function call.}}
    #         ^^^^^^   ^^^^^^< {{Expression without regular expression features.}}
    changed = re.sub(r"\.\.\.", ";", input)  # Noncompliant
    changed = re.sub(r"\Q...\E", ";", input)  # Noncompliant
    changed = re.sub(r"\\", "It's", input)  # Noncompliant
    changed = re.sub(r"\.", "It's", input)  # Noncompliant
    changed = re.sub(r"!", "It's", input)  # Noncompliant
    changed = re.sub(r"{", "It's", input)  # Noncompliant

def compliant():
    input = "Bob is a Bird... Bob is a Plane... Bob is Superman!"
    changed = re.sub(r"(?i)bird", "It's", input)
    changed = re.sub(r"\w*\sis", "It's", input)
    changed = re.sub(r"\.{3}", ";", input)
    changed = re.sub(r"\w", "It's", input)
    changed = re.sub(r"\s", "It's", input)
    changed = re.sub(someVariable, "It's", input)
    changed = re.sub(r".", "It's", input)
    changed = re.sub(r"$", "It's", input)
    changed = re.sub(r"|", "It's", input)
    changed = re.sub(r"(", "It's", input)
    changed = re.sub(r"()", "It's", input)
    changed = re.sub(r"[", "It's", input)
    changed = re.sub(r"[a-z]]", "It's", input)
    changed = re.sub(r"x{3}", "It's", input)
    changed = re.sub(r"^", "It's", input)
    changed = re.sub(r"?", "It's", input)
    changed = re.sub(r"x?", "It's", input)
    changed = re.sub(r"*", "It's", input)
    changed = re.sub(r"x*", "It's", input)
    changed = re.sub(r"+", "It's", input)
    changed = re.sub(r"x+", "It's", input)
    changed = re.sub(r"[\\]", "It's", input)
    changed = re.sub(r"", "It's", input)
    changed = re.match(r"Bob is", input)

def coverage(input):
    re.match(r"Bob is", input)


