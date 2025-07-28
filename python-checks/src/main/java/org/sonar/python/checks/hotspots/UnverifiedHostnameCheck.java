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
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.function.Predicate;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.TriBool;
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
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.semantic.v2.SymbolV2;
import org.sonar.python.semantic.v2.UsageV2;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.TypeCheckBuilder;

@Rule(key = "S5527")
public class UnverifiedHostnameCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Enable server hostname verification on this SSL/TLS connection.";
  private static final String SECONDARY_OPENSSL = "This context does not perform hostname verification.";
  private static final String SECONDARY_DISABLED = "Hostname verification is disabled here.";

  private static final Set<String> SECURE_BY_DEFAULT = new HashSet<>(Arrays.asList("ssl.create_default_context", "ssl._create_default_https_context"));
  private static final Set<String> UNSECURE_BY_DEFAULT = new HashSet<>(Arrays.asList("ssl._create_unverified_context", "ssl._create_stdlib_context"));

  private static final Set<String> UNSAFE_PROTOCOLS = Set.of(
    "ssl.PROTOCOL_SSLv23",
    "ssl.PROTOCOL_TLS",
    "ssl.PROTOCOL_SSLv3",
    "ssl.PROTOCOL_TLSv1",
    "ssl.PROTOCOL_TLSv1_1",
    "ssl.PROTOCOL_TLSv1_2"
  );

  private static Set<String> functionsToCheck;

  private TypeCheckBuilder openSSLConnectionTypeCheckBuilder;
  private TypeCheckBuilder openSSLContextTypeCheckBuilder;
  private TypeCheckBuilder sslContextTypeCheckBuilder;

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
        if (qualifiedExpression != null && "check_hostname".equals(qualifiedExpression.name().name())) {
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
      sslContextTypeCheckBuilder = ctx.typeChecker().typeCheckBuilder().isTypeWithName("ssl.SSLContext");
    });
  }

  private void checkCallExpression(SubscriptionContext ctx) {
    CallExpression callExpression = (CallExpression) ctx.syntaxNode();
    Symbol calleeSymbol = callExpression.calleeSymbol();
    if (calleeSymbol != null && functionsToCheck().contains(calleeSymbol.fullyQualifiedName())) {
      checkSuspiciousCall(callExpression, calleeSymbol, ctx);
    }
    checkOpenSSLConnection(ctx, callExpression);
    checkSSLContext(ctx, callExpression);
  }

  private void checkSSLContext(SubscriptionContext ctx, CallExpression callExpression) {
    Expression callee = callExpression.callee();
    PythonType pythonType = callee.typeV2();

    if (sslContextTypeCheckBuilder.check(pythonType) != TriBool.TRUE) {
      return;
    }

    Optional<Symbol> protocolSymbolOpt = getProtocolSymbol(callExpression);
    if (protocolSymbolOpt.isEmpty()) {
      return;
    }

    Symbol protocolSymbol = protocolSymbolOpt.get();
    String protocolFQN = protocolSymbol.fullyQualifiedName();
    if (protocolFQN == null) {
      return;
    }

    Tree parent = TreeUtils.firstAncestorOfKind(callExpression, Tree.Kind.ASSIGNMENT_STMT);
    if (parent == null) {
      // Direct usage case where context is created and used without assignment
      if (UNSAFE_PROTOCOLS.contains(protocolFQN)) {
        ctx.addIssue(callExpression, MESSAGE);
      }
      return;
    }

    Optional<SymbolV2> contextSymbol = getContextSymbol((AssignmentStatement) parent);
    if (contextSymbol.isEmpty()) {
      return;
    }

    if (UNSAFE_PROTOCOLS.contains(protocolFQN)) {
      // For unsafe protocols, check_hostname should explicitly be set to true
      Optional<AssignmentStatement> hostnameEnabledAssignment = findCheckHostnameStatement(contextSymbol.get(), Expressions::isTruthy);
      if (hostnameEnabledAssignment.isEmpty()) {
        ctx.addIssue(callExpression, MESSAGE);
      }
    } else {
      // For safe protocols, report if check_hostname is explicitly set to false
      Optional<AssignmentStatement> hostnameDisabledAssignment = findCheckHostnameStatement(contextSymbol.get(), Expressions::isFalsy);
      hostnameDisabledAssignment.ifPresent(assignment -> {
        PreciseIssue issue = ctx.addIssue(callee, MESSAGE);
        issue.secondary(assignment, SECONDARY_DISABLED);
      });
    }
  }

  private static Optional<Symbol> getProtocolSymbol(CallExpression callExpression) {
    return Optional.ofNullable(TreeUtils.nthArgumentOrKeyword(0, "protocol", callExpression.arguments()))
      .map(RegularArgument::expression)
      .filter(HasSymbol.class::isInstance)
      .map(expr -> ((HasSymbol) expr).symbol());
  }

  private static Optional<SymbolV2> getContextSymbol(AssignmentStatement assignmentStatement) {
    return Optional.ofNullable(assignmentStatement.lhsExpressions().get(0).expressions().get(0))
      .filter(Name.class::isInstance)
      .map(expr -> ((Name) expr).symbolV2());
  }

  private static Optional<AssignmentStatement> findCheckHostnameStatement(SymbolV2 contextSymbol, Predicate<Expression> valueCheck) {
    return contextSymbol.usages().stream()
      .map(UsageV2::tree)
      .map(Tree::parent)
      .filter(QualifiedExpression.class::isInstance)
      .map(QualifiedExpression.class::cast)
      .filter(qe -> "check_hostname".equals(qe.name().name()))
      .map(t -> TreeUtils.firstAncestorOfKind(t, Tree.Kind.ASSIGNMENT_STMT))
      .filter(Objects::nonNull)
      .map(AssignmentStatement.class::cast)
      .filter(a -> valueCheck.test(a.assignedValue()))
      .findFirst();
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

