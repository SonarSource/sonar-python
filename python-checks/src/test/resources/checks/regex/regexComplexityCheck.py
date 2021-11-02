import re


def character_classes(input):
    re.match(r'[a][b][c][d][e][f][g][h][i][j][k][l][m][n][o][p][q][r][s][t]', input)
    # Noncompliant@+1 {{Simplify this regular expression to reduce its complexity from 21 to the 20 allowed.}}
    re.match(r'[a][b][c][d][e][f][g][h][i][j][k][l][m][n][o][p][q][r][s][t][u]', input)


def disjunction(input):
    # Noncompliant@+1 {{Simplify this regular expression to reduce its complexity from 21 to the 20 allowed.}}
    re.match(r'(a|(b|(c|(d|(e|(f|(gh)))))))', input)  # 1+2+3+4+5+6=21
    # Noncompliant@+1 {{Simplify this regular expression to reduce its complexity from 21 to the 20 allowed.}}
    re.match(r'(a|(b|(c|(d|(e|(((f|(gh)))))))))', input)  # 1+2+3+4+5+6=21
    # Noncompliant@+1 {{Simplify this regular expression to reduce its complexity from 23 to the 20 allowed.}}
    re.match(r'(a|(b|(c|(d|(e|(f|g|h|i))))))', input)  # 1+2+3+4+5+8=23
    # Noncompliant@+1 {{Simplify this regular expression to reduce its complexity from 28 to the 20 allowed.}}
    re.match(r'(a|(b|(c|(d|(e|(f|(g|(hi))))))))', input)  # 1+2+3+4+5+6+7=28


def repetition(input):
    # Noncompliant@+1 {{Simplify this regular expression to reduce its complexity from 21 to the 20 allowed.}}
    re.match(r'(a(b(c(d(ef+)+)+)+)+)+', input)  # 6+5+4+3+2+1=21
    # Noncompliant@+1 {{Simplify this regular expression to reduce its complexity from 21 to the 20 allowed.}}
    re.match(r'(a(b(c(d(ef*)*)*)*)*)*', input)  # 6+5+4+3+2+1=21


def non_capturing_group(input):
    re.match(r'(?:a(?:b(?:c(?:d(?:e(?:f))))))', input)  # 0
    # Noncompliant@+1 {{Simplify this regular expression to reduce its complexity from 21 to the 20 allowed.}}
    re.match(r'(?i:a(?i:b(?i:c(?i:d(?i:e(?i:f))))))', input)  # 1+2+3+4+5+6=21
    # Noncompliant@+1 {{Simplify this regular expression to reduce its complexity from 21 to the 20 allowed.}}
    re.match(r'(?i:a(?i:b(?i:c(?i:d(?i:e((?i)f))))))', input)  # 1+2+3+4+5+6=21
    # Noncompliant@+1 {{Simplify this regular expression to reduce its complexity from 21 to the 20 allowed.}}
    re.match(r'(?-i:a(?-i:b(?-i:c(?-i:d(?-i:e(?-i:f))))))', input)  # 1+2+3+4+5+6=21


def back_reference(input):
    # Noncompliant@+1 {{Simplify this regular expression to reduce its complexity from 21 to the 20 allowed.}}
    re.match(r'(abc)(a|(b|(c|(d|(e|(f|gh))))))', input)  # 1+2+3+4+5+6=21
    # Noncompliant@+1 {{Simplify this regular expression to reduce its complexity from 22 to the 20 allowed.}}
    re.match(r'(abc)(a|(b|(c|(d|(e|(f|\1))))))', input)  # 1+2+3+4+5+6+1=22


def look_around(input):
    # Noncompliant@+1 {{Simplify this regular expression to reduce its complexity from 21 to the 20 allowed.}}
    re.match(r'(a|(b|(c|(d|(e|(f(?!g)))))))', input)  # 1+2+3+4+5+6=21
    # Noncompliant@+1 {{Simplify this regular expression to reduce its complexity from 21 to the 20 allowed.}}
    re.match(r'(a|(b|(c|(d|(e(?!(f|g)))))))', input)  # 1+2+3+4+5+6=21
