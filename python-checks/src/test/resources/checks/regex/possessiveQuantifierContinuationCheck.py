import re

match1 = re.match(r"a++abc", "aaaabc", re.DOTALL) # Noncompliant {{Change this impossible to match sub-pattern that conflicts with the previous possessive quantifier.}}
match2 = re.match(r"\d*+[02468]", "1234", re.DOTALL) # Noncompliant {{Change this impossible to match sub-pattern that conflicts with the previous possessive quantifier.}}

match3 = re.match(r"aa++bc", "aaaabc", re.DOTALL) # Compliant, for example it can match "aaaabc"
match4 = re.match(r"\d*+(?<=[02468])", "1234", re.DOTALL) # Compliant, for example, it can match an even number like "1234"
