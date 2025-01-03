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
import java.util.List;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.NumericLiteral;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S2612")
public class FilePermissionsCheck extends PythonSubscriptionCheck {

  private static final List<String> CHMOD_FUNCTIONS = Arrays.asList("os.chmod", "os.lchmod", "os.fchmod");
  private static final String UMASK_FUNCTION = "os.umask";
  private static final List<String> SENSITIVE_CONSTANTS = Arrays.asList("stat.S_IRWXO", "stat.S_IROTH", "stat.S_IWOTH", "stat.S_IXOTH");
  private static final int CHMOD_MODE_ARG_POSITION = 1;
  private static final int UMASK_MODE_ARG_POSITION = 0;
  private static final int SAFE_CHMOD_MODULO = 0;
  private static final int SAFE_UMASK_MODULO = 7;

  private static final String MESSAGE = "Make sure this permission is safe.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, FilePermissionsCheck::checkCallExpression);
  }

  private static void checkCallExpression(SubscriptionContext ctx) {
    CallExpression callExpression = (CallExpression) ctx.syntaxNode();
    Symbol calleeSymbol = callExpression.calleeSymbol();
    if (calleeSymbol == null) {
      return;
    }
    String calleeFQN = calleeSymbol.fullyQualifiedName();
    List<Argument> arguments = callExpression.arguments();
    if (CHMOD_FUNCTIONS.contains(calleeFQN) && arguments.size() >= 2) {
      checkSensitiveArgument(arguments, CHMOD_MODE_ARG_POSITION, SAFE_CHMOD_MODULO, ctx);
    }
    if (UMASK_FUNCTION.equals(calleeFQN) && arguments.size() == 1) {
      checkSensitiveArgument(arguments, UMASK_MODE_ARG_POSITION, SAFE_UMASK_MODULO, ctx);
    }
  }

  private static void checkSensitiveArgument(List<Argument> arguments , int sensitiveArgPosition, int safeModulo, SubscriptionContext ctx) {
    RegularArgument modeArgument = TreeUtils.nthArgumentOrKeyword(sensitiveArgPosition, "mode", arguments);
    if (modeArgument == null) {
      return;
    }
    Expression expression = modeArgument.expression();
    if(isUnsafeExpression(expression, safeModulo, new HashSet<>())) {
      ctx.addIssue(modeArgument, MESSAGE);
    }
  }

  private static boolean isUnsafeExpression(Expression expression, int safeModulo, Set<Expression> checkedExpressions) {
    if (checkedExpressions.contains(expression)) {
      return false;
    }
    checkedExpressions.add(expression);
    if (expression instanceof HasSymbol hasSymbol) {
      Symbol symbol = hasSymbol.symbol();
      if (symbol != null && SENSITIVE_CONSTANTS.contains(symbol.fullyQualifiedName())) {
        return true;
      }
    }
    if (expression.is(Tree.Kind.BITWISE_OR)) {
      BinaryExpression binaryExpression = (BinaryExpression) expression;
      return isUnsafeExpression(binaryExpression.leftOperand(), safeModulo, checkedExpressions)
        || isUnsafeExpression(binaryExpression.rightOperand(), safeModulo, checkedExpressions);
    }
    if (expression.is(Tree.Kind.NUMERIC_LITERAL)) {
      NumericLiteral numericLiteral = (NumericLiteral) expression;
      return numericLiteral.valueAsLong() % 8 != safeModulo;
    }
    if (expression.is(Tree.Kind.NAME)) {
      Expression singleAssignedValue = Expressions.singleAssignedValue(((Name) expression));
      if (singleAssignedValue == null) {
        return false;
      }
      return isUnsafeExpression(singleAssignedValue, safeModulo, checkedExpressions);
    }
    return false;
  }
}
