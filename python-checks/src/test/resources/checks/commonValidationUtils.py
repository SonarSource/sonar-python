callExpr(10)
callExpr(9)  # Noncompliant

var_nok = 9
var_ok = 10
callExpr(var_ok)
callExpr(var_nok)  # Noncompliant

var_wrong = "something"
callExpr(var_wrong)

callExpr(12, 0)
callExpr(12, 10)  # Noncompliant {{Argument is equal to 10}}
callExpr(12, 10.0)  # Noncompliant {{Argument is equal to 10}}
ten = 10
callExpr(isEqualTo=ten)  # Noncompliant {{Argument is equal to 10}}
not_ten = 11
callExpr(isEqualTo=not_ten)
callExpr(12, var_wrong)
