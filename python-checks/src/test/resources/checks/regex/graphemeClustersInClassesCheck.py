import re


def non_compliant(input):
    re.match(r'[aaaèaaa]', input)  # Noncompliant {{Extract 1 Grapheme Cluster(s) from this character class.}}
    #          ^^^^^^^^^^
    re.match(r'[0Ṩ0]', input)  # Noncompliant {{Extract 1 Grapheme Cluster(s) from this character class.}}
    re.match(r'aaa[è]aaa', input)  # Noncompliant
    # two secondary per line: one for the regex location, and one for the cluster location
    re.match(r'[èaaèaaè]', input)  # Noncompliant {{Extract 3 Grapheme Cluster(s) from this character class.}}
    re.match(r'[èa-dä]', input)  # Noncompliant
    re.match(r'[èa]aaa[dè]', input)     # Noncompliant 2
    re.match(r'[ä]', input)  # Noncompliant
    re.match(r'[c̈]', input)  # Noncompliant
    re.match(r'[e⃝]', input)  # Noncompliant


def compliant(input):
    re.match(r'[é]', input)  # Compliant, a single char
    re.match(r'[e\u0300]', input)  # Compliant, escaped unicode
    re.match(r'[e\x{0300}]', input)  # Compliant, escaped unicode
    re.match(r'[e\u20DD̀]', input)  # Compliant, (letter, escaped unicode, mark) can not be combined
    re.match(r'[\u0300e]', input)  # Compliant, escaped unicode, letter
    re.match(r'[̀̀]', input)  # Compliant, two marks
    re.match(r'[̀̀]', input)  # Compliant, one mark

    re.match(r'/ä/', input) # Compliant, not in a class
