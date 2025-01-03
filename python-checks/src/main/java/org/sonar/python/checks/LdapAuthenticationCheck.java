/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource SA.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.python.checks;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.ArgList;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.BuiltinTypes;
import org.sonar.python.checks.utils.Expressions;

import static org.sonar.plugins.python.api.tree.Tree.Kind.CALL_EXPR;
import static org.sonar.plugins.python.api.tree.Tree.Kind.NAME;
import static org.sonar.plugins.python.api.tree.Tree.Kind.STRING_LITERAL;
import static org.sonar.plugins.python.api.tree.Tree.Kind.UNPACKING_EXPR;

@Rule(key = "S4433")
public class LdapAuthenticationCheck extends PythonSubscriptionCheck {

  private static final Set<String> LDAP_OBJECT_SENSITIVE_METHODS = new HashSet<>(
    Arrays.asList("ldap.ldapobject.SimpleLDAPObject.simple_bind", "ldap.ldapobject.SimpleLDAPObject.simple_bind_s",
      "ldap.ldapobject.SimpleLDAPObject.bind", "ldap.ldapobject.SimpleLDAPObject.bind_s"));

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(CALL_EXPR, ctx -> {
      CallExpression callExpression = (CallExpression) ctx.syntaxNode();
      Symbol calleeSymbol = callExpression.calleeSymbol();
      Set<Tree> secondaries = new HashSet<>();
      if (calleeSymbol != null && LDAP_OBJECT_SENSITIVE_METHODS.contains(calleeSymbol.fullyQualifiedName()) && !isPasswordProvided(callExpression.argumentList(), secondaries)) {
        PreciseIssue preciseIssue = ctx.addIssue(callExpression.callee(), "Provide a password when authenticating to this LDAP server.");
        secondaries.forEach(secondary -> preciseIssue.secondary(secondary, null));
      }
    });
  }

  private static boolean isPasswordProvided(@Nullable ArgList argList, Set<Tree> secondaries) {
    if (argList == null) {
      return false;
    }
    for (int i = 0; i < argList.arguments().size(); i++) {
      if (argList.arguments().get(i).is(UNPACKING_EXPR)) {
        return true;
      }
      RegularArgument regularArgument = (RegularArgument) argList.arguments().get(i);
      Name keyword = regularArgument.keywordArgument();
      if ((keyword == null && i == 1) ||
        (keyword != null && keyword.name().equals("cred"))) {

        if (isValidPassword(regularArgument.expression(), secondaries)) {
          return true;
        } else {
          secondaries.add(regularArgument.expression());
          return false;
        }
      }
    }
    return false;
  }

  private static boolean isValidPassword(Expression expression, Set<Tree> secondaries) {
    if (isNoneOrEmptyString(expression, secondaries)) {
      return false;
    }
    if (expression.is(NAME)) {
      Expression singleAssignedValue = Expressions.singleAssignedValue(((Name) expression));
      if (singleAssignedValue != null) {
        return !isNoneOrEmptyString(singleAssignedValue, secondaries);
      }
    }
    return true;
  }

  private static boolean isNoneOrEmptyString(Expression expression, Set<Tree> secondaries) {
    if (expression.type().canOnlyBe(BuiltinTypes.NONE_TYPE) ||
      (expression.is(STRING_LITERAL) && ((StringLiteral) expression).trimmedQuotesValue().isEmpty())) {

      secondaries.add(expression);
      return true;
    }
    return false;
  }
}
