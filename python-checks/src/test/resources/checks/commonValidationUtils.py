callExpr(10)
callExpr(9)  # Noncompliant

var_nok = 9
var_ok = 10
callExpr(var_ok)
callExpr(var_nok)  # Noncompliant

var_wrong = "something"
callExpr(var_wrong)
