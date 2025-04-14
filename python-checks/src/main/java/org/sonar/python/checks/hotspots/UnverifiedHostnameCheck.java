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
package org.sonar.python.checks.hotspots;

import java.util.Arrays;
import java.util.Collections;
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
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.TriBool;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.TypeCheckBuilder;

@Rule(key = "S5527")
public class UnverifiedHostnameCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Enable server hostname verification on this SSL/TLS connection.";
  private static final String SECONDARY_OPENSSL = "This context does not perform hostname verification.";

  private static final Set<String> SECURE_BY_DEFAULT = new HashSet<>(Arrays.asList("ssl.create_default_context", "ssl._create_default_https_context"));
  private static final Set<String> UNSECURE_BY_DEFAULT = new HashSet<>(Arrays.asList("ssl._create_unverified_context", "ssl._create_stdlib_context"));

  private static Set<String> functionsToCheck;

  private TypeCheckBuilder openSSLConnectionTypeCheckBuilder;
  private TypeCheckBuilder openSSLContextTypeCheckBuilder;

  private static Set<String> functionsToCheck() {
    if (functionsToCheck == null) {
      functionsToCheck = new HashSet<>();
      functionsToCheck.addAll(SECURE_BY_DEFAULT);
      functionsToCheck.addAll(UNSECURE_BY_DEFAULT);
    }
    return Collections.unmodifiableSet(functionsToCheck);
  }

  private static void checkSuspiciousCall(CallExpression callExpression, Symbol calleeSymbol, SubscriptionContext ctx) {
    Tree parent = TreeUtils.firstAncestorOfKind(callExpression, Tree.Kind.ASSIGNMENT_STMT, Tree.Kind.CALL_EXPR);
    if (parent == null) {
      return;
    }
    if (parent.is(Tree.Kind.ASSIGNMENT_STMT)) {
      Expression lhs = ((AssignmentStatement) parent).lhsExpressions().get(0).expressions().get(0);
      if (lhs instanceof HasSymbol hasSymbol) {
        Symbol symbol = hasSymbol.symbol();
        if (symbol == null) {
          return;
        }
        if (isUnsafeContext(calleeSymbol, symbol)) {
          ctx.addIssue(callExpression, MESSAGE);
        }
      }
    } else if (opensUnsecureConnection(calleeSymbol, (CallExpression) parent)) {
      ctx.addIssue(callExpression, MESSAGE);
    }
  }

  private static boolean isUnsafeContext(Symbol calleeSymbol, Symbol symbol) {
    for (Usage usage : symbol.usages()) {
      if (usage.kind().equals(Usage.Kind.OTHER)) {
        QualifiedExpression qualifiedExpression = (QualifiedExpression) TreeUtils.firstAncestorOfKind(usage.tree(), Tree.Kind.QUALIFIED_EXPR);
        if (qualifiedExpression != null && qualifiedExpression.name().name().equals("check_hostname")) {
          AssignmentStatement assignmentStatement = (AssignmentStatement) TreeUtils.firstAncestorOfKind(qualifiedExpression, Tree.Kind.ASSIGNMENT_STMT);
          if (assignmentStatement != null) {
            return Expressions.isFalsy(assignmentStatement.assignedValue());
          }
        }
      }
    }
    return UNSECURE_BY_DEFAULT.contains(calleeSymbol.fullyQualifiedName());
  }

  private static boolean opensUnsecureConnection(Symbol calleeSymbol, CallExpression callExpr) {
    Symbol parentCalleeSymbol = callExpr.calleeSymbol();
    return parentCalleeSymbol != null && "urllib.request.urlopen".equals(parentCalleeSymbol.fullyQualifiedName())
      && UNSECURE_BY_DEFAULT.contains(calleeSymbol.fullyQualifiedName());
  }

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, this::checkCallExpression);
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> {
      openSSLConnectionTypeCheckBuilder = ctx.typeChecker().typeCheckBuilder().isTypeWithName("OpenSSL.SSL.Connection");
      openSSLContextTypeCheckBuilder = ctx.typeChecker().typeCheckBuilder().isInstanceOf("OpenSSL.SSL.Context");
    });
  }

  private void checkCallExpression(SubscriptionContext ctx) {
    CallExpression callExpression = (CallExpression) ctx.syntaxNode();
    Symbol calleeSymbol = callExpression.calleeSymbol();
    if (calleeSymbol != null && functionsToCheck().contains(calleeSymbol.fullyQualifiedName())) {
      checkSuspiciousCall(callExpression, calleeSymbol, ctx);
    }
    checkOpenSSLConnection(ctx, callExpression);
  }

  private void checkOpenSSLConnection(SubscriptionContext ctx, CallExpression callExpression) {
    Expression callee = callExpression.callee();
    PythonType pythonType = callee.typeV2();

    if (openSSLConnectionTypeCheckBuilder.check(pythonType) != TriBool.TRUE) {
      return;
    }

    RegularArgument contextArg = TreeUtils.nthArgumentOrKeyword(0, "context", callExpression.arguments());
    if (contextArg == null) {
      return;
    }

    Expression contextExpr = contextArg.expression();
    PythonType contextType = contextExpr.typeV2();
    if (openSSLContextTypeCheckBuilder.check(contextType) == TriBool.TRUE) {
      PreciseIssue issue = ctx.addIssue(callee, MESSAGE);
      Expressions.ifNameGetSingleAssignedNonNameValue(contextExpr)
        .ifPresentOrElse(e -> issue.secondary(e, SECONDARY_OPENSSL),
          () -> issue.secondary(contextExpr, SECONDARY_OPENSSL)
        );
    }
  }
}
