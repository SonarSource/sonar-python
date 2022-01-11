import re


def non_compliant(input):
    re.match(r"[0-99]", input)  # Noncompliant {{Remove duplicates in this character class.}}
    #           ^^^
    #              ^@-1< {{Additional duplicate}}
    re.match(r"[90-9]", input)  # Noncompliant
    re.match(r"[0-73-9]", input)  # Noncompliant
    re.match(r"[0-93-57]", input)  # Noncompliant
    re.match(r"[4-92-68]", input)  # Noncompliant
    re.match(r"[0-33-9]", input)  # Noncompliant
    re.match(r"[0-70-9]", input)  # Noncompliant
    re.match(r"[3-90-7]", input)  # Noncompliant
    re.match(r"[3-50-9]", input)  # Noncompliant
    re.match(r"[xxx]", input)  # Noncompliant
    re.match(r"[A-z_]", input)  # Noncompliant
    re.match(r"(?i)[A-Za-z]", input)  # Noncompliant
    re.match(r"(?i)[A-_d]", input)  # Noncompliant
    re.match(r"(?iu)[Ä-Üä]", input)  # Noncompliant
    re.match(r"(?iu)[a-Öö]", input)  # Noncompliant
    re.match(r"[  ]", input)  # Noncompliant
    re.match(r"(?i)[  ]", input)  # Noncompliant
    re.match(r"(?iu)[  ]", input)  # Noncompliant
    re.match(r"(?i)[A-_D]", input)  # Noncompliant
    re.match(r"(?iu)[A-_D]", input)  # Noncompliant
    re.match(r"(?i)[xX]", input)  # Noncompliant
    re.match(r"(?iu)[äÄ]", input)  # Noncompliant
    re.match(r"(?iU)[äÄ]", input)  # Noncompliant
    re.match(r"(?iu)[xX]", input)  # Noncompliant
    re.match(r"[\"\".]", input)  # Noncompliant
    re.match(r"[\x{F600}-\x{F637}\x{F608}]", input)  # Noncompliant
    re.match(r"[\Qxx\E]", input)  # Noncompliant
    re.match(r"[\s\Sx]", input)  # Noncompliant
    re.match(r"(?U)[\s\Sx]", input)  # Noncompliant
    re.match(r"[\w\d]", input)  # Noncompliant
    re.match(r"[\wa]", input)  # Noncompliant
    re.match(r"[\d1]", input)  # Noncompliant
    re.match(r"[\d1-3]", input)  # Noncompliant
    re.match(r"(?U)[\wa]", input)  # Noncompliant
    re.match(r"[A-Za-z]", input, re.IGNORECASE)  # Noncompliant
    re.match(r"[0-9\d]", input)  # Noncompliant
    re.match(r"[0-9\d]", input)  # Noncompliant
    re.match(r"[0-9\\\d]", input)  # Noncompliant
    re.match(r"(?(?=1)[0-99])", input)  # Noncompliant
    re.match(r"(?(?=1)1|[0-99])", input)  # Noncompliant
    # UNICODE flag is always enabled
    re.match(r"(?i)[äÄ]", input) # Noncompliant
    re.match(r"(?i)[Ä-Üä]", input) # Noncompliant
    re.match(r"(?i)[a-Öö]", input) # Noncompliant
    re.match(r"[[^\s\S]x]", input) # Noncompliant
    re.match(r"(?U)[[^\W]a]", input)  # Noncompliant


def compliant(input):
    re.match(r"a-z\d", input)
    re.match(r"[0-9][0-9]?", input)
    re.match(r"[xX]", input)
    re.match(r"[\s\S]", input)
    re.match(r"(?U)[\s\S]", input)
    re.match(r"(?U)[\S\u0085\u2028\u2029]", input)
    re.match(r"[\d\D]", input)
    re.match(r"(?U)[\d\D]", input)
    re.match(r"[\w\W]", input)
    re.match(r"(?U)[\w\W]", input)
    re.match(r"[\wä]", input)
    re.match(r"(?i)[äÄ]", input, re.ASCII)
    re.match(r"(?i)[Ä-Üä]", input, re.ASCII)
    re.match(r"(?u)[äÄ]", input)
    re.match(r"(?u)[xX]", input)
    re.match(r"[ab-z]", input)
    re.match(r"[[a][b]]", input)
    re.match(r"[[^a]a]", input)
    re.match(r"[Z-ax]", input, re.IGNORECASE)
    re.match(r"(?i)[a-Öö]", input, re.ASCII)
    re.match(r"[0-9\Q.-_\E]", input)  # This used to falsely interpret .-_ as a range and complain that it overlaps with 0-9
    re.match(r"[A-Z\Q-_.\E]", input)
    re.match(r"[\x00\x01]]", input)  # This used to falsely complain about x and 0 being duplicates
    re.match(r"[\x00-\x01\x02-\x03]]", input)
    re.match(r"[z-a9-0]", input)  # Illegal character class should not make the check explode
    re.match(r"[aa", input)  # Check should not run on syntactically invalid regexen
    re.match(r"(?U)[\wä]", input)  # False negative because we don't support Unicode characters in \w and \W
    re.match(r"[[a-z&&b-e]c]", input)  # FN because we don't support intersections
    re.match(r"(?i)[A-_d-{]", input)  # Noncompliant
    re.match(r"(?i)[A-z_]", input)  # Noncompliant
    re.match(r"[\abc]", input)
    re.match(r'[\s\'"\:\{\}\[\],&\*\#\?]', input)
    re.match(r"[0-9\\d]", input)  # Compliant


def emoji(input):
    re.match(r"[😂😊]", input)  # Compliant
    re.match(r"[^\ud800\udc00-\udbff\udfff]", input)  # Compliant
