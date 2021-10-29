import re


def non_compliant(input):
    re.match(r"<.+?>", input)  # Noncompliant {{Replace this use of a reluctant quantifier with "[^>]++".}}
    re.match(r"<\S+?>", input)  # Noncompliant {{Replace this use of a reluctant quantifier with "[^>\s]++".}}
    re.match(r"<\D+?>", input)  # Noncompliant {{Replace this use of a reluctant quantifier with "[^>\d]++".}}
    re.match(r"<\W+?>", input)  # Noncompliant {{Replace this use of a reluctant quantifier with "[^>\w]++".}}

    re.match(r"<.{2,5}?>", input)  # Noncompliant {{Replace this use of a reluctant quantifier with "[^>]{2,5}+".}}
    re.match(r"<\S{2,5}?>", input)  # Noncompliant {{Replace this use of a reluctant quantifier with "[^>\s]{2,5}+".}}
    re.match(r"<\D{2,5}?>", input)  # Noncompliant {{Replace this use of a reluctant quantifier with "[^>\d]{2,5}+".}}
    re.match(r"<\W{2,5}?>", input)  # Noncompliant {{Replace this use of a reluctant quantifier with "[^>\w]{2,5}+".}}

    re.match(r"<.{2,}?>", input)  # Noncompliant {{Replace this use of a reluctant quantifier with "[^>]{2,}+".}}
    re.match(r"\".*?\"", input)  # Noncompliant {{Replace this use of a reluctant quantifier with "[^\"]*+".}}
    re.match(r".*?\w", input)  # Noncompliant {{Replace this use of a reluctant quantifier with "\W*+".}}
    re.match(r".*?\W", input)  # Noncompliant {{Replace this use of a reluctant quantifier with "\w*+".}}
    re.match(r".*?\p{L}", input)  # Noncompliant {{Replace this use of a reluctant quantifier with "\P{L}*+".}}
    re.match(r".*?\P{L}", input)  # Noncompliant {{Replace this use of a reluctant quantifier with "\p{L}*+".}}
    re.match(r"\[.*?\]", input)  # Noncompliant {{Replace this use of a reluctant quantifier with "[^\]]*+".}}
    re.match(r".+?[abc]", input)  # Noncompliant {{Replace this use of a reluctant quantifier with "[^abc]++".}}
    re.match(r"(?-U:\s)*?\S", input)
    re.match(r"(?U:\s)*?\S", input)  # Noncompliant {{Replace this use of a reluctant quantifier with "[\s\S]*+".}}
    re.match(r"(?U:a|\s)*?\S", input)
    re.match(r"\S*?\s", input)
    re.match(r"\S*?(?-U:\s)", input)
    re.match(r"\S*?(?U:\s)", input)  # Noncompliant {{Replace this use of a reluctant quantifier with "[\S\s]*+".}}
    re.match(r"\S*?(?U)\s", input)  # Noncompliant {{Replace this use of a reluctant quantifier with "[\S\s]*+".}}

    # coverage
    re.match(r"(?:(?m))*?a", input)
    re.match(r"(?:(?m:.))*?(?:(?m))", input)

    # This replacement might not be equivalent in case of full match, but is equivalent in case of split
    re.match(r".+?[^abc]", input)  # Noncompliant {{Replace this use of a reluctant quantifier with "[abc]++".}}

    re.match(r".+?\x{1F4A9}", input)  # Noncompliant {{Replace this use of a reluctant quantifier with "[^\x{1F4A9}]++".}}
    re.match(r"<abc.*?>", input)  # Noncompliant {{Replace this use of a reluctant quantifier with "[^>]*+".}}
    re.match(r"<.+?>|otherstuff", input)  # Noncompliant {{Replace this use of a reluctant quantifier with "[^>]++".}}
    re.match(r"(<.+?>)*", input)  # Noncompliant {{Replace this use of a reluctant quantifier with "[^>]++".}}

    re.match(r"\S+?[abc]", input)  # Noncompliant {{Replace this use of a reluctant quantifier with "[^abc\s]++".}}
    re.match(r"\D+?[abc]", input)  # Noncompliant {{Replace this use of a reluctant quantifier with "[^abc\d]++".}}
    re.match(r"\w+?[abc]", input)  # Noncompliant {{Replace this use of a reluctant quantifier with "[^abc\W]++".}}

    re.match(r"\S*?[abc]", input)  # Noncompliant {{Replace this use of a reluctant quantifier with "[^abc\s]*+".}}
    re.match(r"\D*?[abc]", input)  # Noncompliant {{Replace this use of a reluctant quantifier with "[^abc\d]*+".}}
    re.match(r"\w*?[abc]", input)  # Noncompliant {{Replace this use of a reluctant quantifier with "[^abc\W]*+".}}

    re.match(r"\S+?[^abc]", input)  # Noncompliant {{Replace this use of a reluctant quantifier with "[abc\S]++".}}
    re.match(r"\s+?[^abc]", input)  # Noncompliant {{Replace this use of a reluctant quantifier with "[abc\s]++".}}


def compliant(input):
    re.match(r"<[^>]++>", input)
    re.match(r"<[^>]+>", input)
    re.match(r"<[^>]+?>", input)
    re.match(r"<.{42}?>", input)  # Adding a ? to a fixed quantifier is pointless, but also doesn't cause any backtracking issues
    re.match(r"<.+>", input)
    re.match(r"<.++>", input)
    re.match(r"<--.?-->", input)
    re.match(r"<--.+?-->", input)
    re.match(r"<--.*?-->", input)
    re.match(r"/\*.?\*/", input)
    re.match(r"<[^>]+>?", input)
    re.match(r"", input)
    re.match(r".*?(?:a|b|c)", input)  # Alternatives are currently not covered even if they contain only single characters


def no_intersection(input):
    re.match(r"<\d+?>", input)
    re.match(r"<\s+?>", input)
    re.match(r"<\w+?>", input)

    re.match(r"<\s{2,5}?>", input)
    re.match(r"<\d{2,5}?>", input)
    re.match(r"<\w{2,5}?>", input)

    re.match(r"\d+?[abc]", input)
    re.match(r"\s+?[abc]", input)
    re.match(r"\W+?[abc]", input)

    re.match(r"\W*?[abc]", input)
    re.match(r"\s*?[abc]", input)
    re.match(r"\d*?[abc]", input)
