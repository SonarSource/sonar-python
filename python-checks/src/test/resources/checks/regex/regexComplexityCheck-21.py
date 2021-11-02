import re


def character_classes(input):
    re.match(r'[a][b][c][d][e][f][g][h][i][j][k][l][m][n][o][p][q][r][s][t][u]', input)
    # Noncompliant@+1 {{Simplify this regular expression to reduce its complexity from 22 to the 21 allowed.}}
    re.match(r'[a][b][c][d][e][f][g][h][i][j][k][l][m][n][o][p][q][r][s][t][u][v]', input)
