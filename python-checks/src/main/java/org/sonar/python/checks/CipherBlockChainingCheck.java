/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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

import java.util.HashSet;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.TreeUtils;

import static java.util.Arrays.asList;

@Rule(key = "S3329")
public class CipherBlockChainingCheck extends PythonSubscriptionCheck {

  private static final HashSet<String> PYCRYPTO_SENSITIVE_FQNS = new HashSet<>();
  private static final String CRYPTOGRAPHY_SENSITIVE_FQN = "cryptography.hazmat.primitives.ciphers.Cipher";
  private static final String MESSAGE = "Use a dynamically-generated, random IV.";

  static {
    for (String libraryName : asList("Cryptodome", "Crypto")) {
      for (String vulnerableMethodName : asList("AES", "ARC2", "Blowfish", "CAST", "DES", "DES3")) {
        PYCRYPTO_SENSITIVE_FQNS.add(String.format("%s.Cipher.%s.new", libraryName, vulnerableMethodName));
      }
    }
  }

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, CipherBlockChainingCheck::checkCallExpression);
  }

  private static void checkCallExpression(SubscriptionContext ctx) {
    CallExpression callExpression = (CallExpression) ctx.syntaxNode();
    Symbol calleeSymbol = callExpression.calleeSymbol();
    if (calleeSymbol == null || calleeSymbol.fullyQualifiedName() == null) {
      return;
    }
    if (PYCRYPTO_SENSITIVE_FQNS.contains(calleeSymbol.fullyQualifiedName())) {
      checkPyCryptoCall(callExpression, ctx);
    }
    if (CRYPTOGRAPHY_SENSITIVE_FQN.equals(calleeSymbol.fullyQualifiedName())) {
      checkCrypographyCall(callExpression, ctx);
    }
  }

  private static void checkCrypographyCall(CallExpression callExpression, SubscriptionContext ctx) {
    RegularArgument modeArgument = TreeUtils.nthArgumentOrKeyword(1, "mode", callExpression.arguments());
    if (modeArgument == null) {
      return;
    }
    Expression modeArgumentExpression = modeArgument.expression();
    if (!modeArgumentExpression.is(Tree.Kind.CALL_EXPR)) {
      return;
    }
    CallExpression modeCallExpression = (CallExpression) modeArgumentExpression;
    Symbol calleeSymbol = modeCallExpression.calleeSymbol();
    if (calleeSymbol == null || !"CBC".equals(calleeSymbol.name())) {
      return;
    }
    RegularArgument initializationVector = TreeUtils.nthArgumentOrKeyword(0, "initialization_vector", modeCallExpression.arguments());
    if (initializationVector == null || !isStaticInitializationVector(initializationVector.expression(), new HashSet<>())) {
      return;
    }
    AssignmentStatement assignmentStatement = (AssignmentStatement) TreeUtils.firstAncestorOfKind(callExpression, Tree.Kind.ASSIGNMENT_STMT);
    if (assignmentStatement == null || assignmentStatement.assignedValue() != callExpression) {
      return;
    }
    checkAssignmentStatement(assignmentStatement, callExpression, "encryptor", ctx);
  }

  private static void checkPyCryptoCall(CallExpression callExpression, SubscriptionContext ctx) {
    RegularArgument modeArgument = TreeUtils.nthArgumentOrKeyword(1, "mode", callExpression.arguments());
    if (modeArgument == null) {
      return;
    }
    Expression modeArgumentExpression = modeArgument.expression();
    if (!(modeArgumentExpression instanceof HasSymbol)) {
      return;
    }
    Symbol symbol = ((HasSymbol) modeArgumentExpression).symbol();
    if (symbol == null || !"MODE_CBC".equals(symbol.name())) {
      return;
    }
    RegularArgument ivArgument = TreeUtils.nthArgumentOrKeyword(2, "iv", callExpression.arguments());
    if (ivArgument == null || !isStaticInitializationVector(ivArgument.expression(), new HashSet<>())) {
      return;
    }
    AssignmentStatement assignmentStatement = (AssignmentStatement) TreeUtils.firstAncestorOfKind(callExpression, Tree.Kind.ASSIGNMENT_STMT);
    if (assignmentStatement == null || assignmentStatement.assignedValue() != callExpression) {
      return;
    }
    checkAssignmentStatement(assignmentStatement, callExpression, "encrypt", ctx);
  }

  private static void checkAssignmentStatement(AssignmentStatement assignmentStatement, CallExpression callExpression, String suspiciousCallee, SubscriptionContext ctx) {
    assignmentStatement.lhsExpressions().stream()
      .filter(exprList -> exprList.expressions().size() == 1)
      .flatMap(exprList -> exprList.expressions().stream())
      .filter(expression -> expression.is(Tree.Kind.NAME))
      .forEach(name -> {
        Symbol symbol = ((Name) name).symbol();
        if (symbol == null) {
          return;
        }
        symbol.usages().stream()
          .map(Usage::tree)
          .filter(t -> isWithinCallTo(t, suspiciousCallee))
          .findFirst()
          .ifPresent(t -> ctx.addIssue(t, MESSAGE).secondary(callExpression, null));
      });
  }

  private static boolean isWithinCallTo(Tree tree, String calleeName) {
    CallExpression callExpression = (CallExpression) TreeUtils.firstAncestorOfKind(tree, Tree.Kind.CALL_EXPR);
    if (callExpression == null) {
      return false;
    }
    return callExpression.callee().is(Tree.Kind.QUALIFIED_EXPR) && ((QualifiedExpression) callExpression.callee()).name().name().equals(calleeName);
  }

  private static boolean isStaticInitializationVector(Expression expression, Set<Expression> checkedExpressions) {
    if (checkedExpressions.contains(expression)) {
      return false;
    }
    checkedExpressions.add(expression);
    if (expression.is(Tree.Kind.CALL_EXPR) || TreeUtils.hasDescendant(expression, tree -> tree.is(Tree.Kind.CALL_EXPR))) {
      return false;
    }
    if (expression.is(Tree.Kind.NAME)) {
      Expression singleAssignedValue = Expressions.singleAssignedValue((Name) expression);
      if (singleAssignedValue == null) {
        return false;
      }
      return isStaticInitializationVector(singleAssignedValue, checkedExpressions);
    }
    return true;
  }
}
