callExpr("abc")  # Noncompliant {{Argument is abc}}
callExpr("def")

abc_var = "abc"
callExpr(abc_var)  # Noncompliant {{Argument is abc}}

def_var = "def"
callExpr(def_var)

callExpr(arg="abc")  # Noncompliant {{Argument is abc}}
callExpr(arg="def")

callExpr(arg=f"{abc_var}")  # FN f-string are not supported
callExpr(arg="def")

callExpr(arg=3)
not_a_string = 3
callExpr(arg=not_a_string)
