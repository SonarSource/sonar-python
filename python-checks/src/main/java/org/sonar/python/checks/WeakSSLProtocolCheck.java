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
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.function.Predicate;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.PythonVersionUtils;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.CompoundAssignmentStatement;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.TriBool;
import org.sonar.python.checks.cdk.WeakSSLProtocolCheckPart;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.semantic.v2.SymbolV2;
import org.sonar.python.semantic.v2.UsageV2;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.TypeCheckBuilder;

@Rule(key = "S4423")
public class WeakSSLProtocolCheck extends PythonSubscriptionCheck {
  private static final List<String> WEAK_PROTOCOL_CONSTANTS = Arrays.asList(
    "ssl.PROTOCOL_SSLv2",
    "ssl.PROTOCOL_SSLv3",
    "ssl.PROTOCOL_SSLv23",
    "ssl.PROTOCOL_TLSv1",
    "ssl.PROTOCOL_TLSv1_1",
    "OpenSSL.SSL.SSLv2_METHOD",
    "OpenSSL.SSL.SSLv3_METHOD",
    "OpenSSL.SSL.SSLv23_METHOD",
    "OpenSSL.SSL.TLSv1_METHOD",
    "OpenSSL.SSL.TLSv1_1_METHOD"
  );

  private static final Set<String> SSL_CONTEXT_DEPENDENT_PROTOCOLS = Set.of(
    "ssl.PROTOCOL_TLS_CLIENT",
    "ssl.PROTOCOL_TLS_SERVER",
    "ssl.PROTOCOL_TLS"
  );

  private static final Set<String> OPENSSL_DEFAULT_TLS_METHODS = Set.of(
    "OpenSSL.SSL.TLS_METHOD",
    "OpenSSL.SSL.TLS_SERVER_METHOD",
    "OpenSSL.SSL.TLS_CLIENT_METHOD"
  );

  private static final Set<String> SAFE_VERSION_NAMES = Set.of(
    "TLSv1_2",
    "TLSv1_3",
    "MAXIMUM_SUPPORTED"
  );

  private static final Set<String> REQUIRED_SECURITY_FLAGS = Set.of(
    "OP_NO_SSLv2",
    "OP_NO_SSLv3",
    "OP_NO_TLSv1",
    "OP_NO_TLSv1_1"
  );

  private static final Set<String> DEFAULT_PURPOSES = Set.of(
    "ssl.Purpose.CLIENT_AUTH",
    "ssl.Purpose.SERVER_AUTH"
  );

  private static final String WEAK_PROTOCOL_MESSAGE = "Change this code to use a stronger protocol.";

  private TypeCheckBuilder createDefaultContextTypeCheckBuilder;
  private TypeCheckBuilder sslSSLContextTypeCheckBuilder;
  private TypeCheckBuilder openSSLContextTypeCheckBuilder;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> {
      createDefaultContextTypeCheckBuilder = ctx.typeChecker().typeCheckBuilder().isTypeWithName("ssl.create_default_context");
      sslSSLContextTypeCheckBuilder = ctx.typeChecker().typeCheckBuilder().isTypeWithName("ssl.SSLContext");
      openSSLContextTypeCheckBuilder = ctx.typeChecker().typeCheckBuilder().isTypeWithName("OpenSSL.SSL.Context");
    });
    context.registerSyntaxNodeConsumer(Tree.Kind.NAME, WeakSSLProtocolCheck::checkName);
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, this::checkCallExpression);

    new WeakSSLProtocolCheckPart().initialize(context);
  }

  private static void checkName(SubscriptionContext ctx) {
    Name name = (Name) ctx.syntaxNode();
    Optional.of(name)
      .map(HasSymbol::symbol)
      .map(Symbol::fullyQualifiedName)
      .filter(WEAK_PROTOCOL_CONSTANTS::contains)
      .ifPresent(fqn -> ctx.addIssue(name, WEAK_PROTOCOL_MESSAGE));
  }

  private void checkCallExpression(SubscriptionContext ctx) {
    CallExpression callExpression = (CallExpression) ctx.syntaxNode();
    Expression callee = callExpression.callee();
    PythonType pythonType = callee.typeV2();
    if (isSslContextWithDefaultProtocols(pythonType, callExpression)) {
      checkSSLContext(ctx, callExpression);
    } else if (isOpenSSLContextWithDefaultTLSMethods(pythonType, callExpression)) {
      checkOpenSSLContext(ctx, callExpression);
    }
  }

  private boolean isSslContextWithDefaultProtocols(PythonType pythonType, CallExpression callExpression) {
    return (createDefaultContextTypeCheckBuilder.check(pythonType) == TriBool.TRUE && hasDefaultFirstArgument(callExpression, "purpose", DEFAULT_PURPOSES))
      || (sslSSLContextTypeCheckBuilder.check(pythonType) == TriBool.TRUE && hasDefaultFirstArgument(callExpression, "protocol", SSL_CONTEXT_DEPENDENT_PROTOCOLS));
  }

  private boolean isOpenSSLContextWithDefaultTLSMethods(PythonType pythonType, CallExpression callExpression) {
    return openSSLContextTypeCheckBuilder.check(pythonType) == TriBool.TRUE && hasDefaultFirstArgument(callExpression, "method", OPENSSL_DEFAULT_TLS_METHODS);
  }

  private static boolean hasDefaultFirstArgument(CallExpression callExpr, String keyword, Set<String> allowedValues) {
    var arg = TreeUtils.nthArgumentOrKeyword(0, keyword, callExpr.arguments());
    if (arg == null) {
      return true;
    }
    return Optional.of(arg.expression())
      .filter(HasSymbol.class::isInstance)
      .map(HasSymbol.class::cast)
      .map(HasSymbol::symbol)
      .map(Symbol::fullyQualifiedName)
      .filter(allowedValues::contains)
      .isPresent();
  }

  private static void checkSSLContext(SubscriptionContext ctx, Tree tree) {
    getContextSymbol(tree)
      .ifPresentOrElse(
        contextSymbol -> checkSSLContextSymbol(ctx, contextSymbol, tree),
        () -> ctx.addIssue(tree, WEAK_PROTOCOL_MESSAGE)
      );
  }

  private static void checkOpenSSLContext(SubscriptionContext ctx, Tree tree) {
    getContextSymbol(tree)
      .ifPresentOrElse(
        contextSymbol -> checkOpenSSLContextSymbol(ctx, contextSymbol, tree),
        () -> ctx.addIssue(tree, WEAK_PROTOCOL_MESSAGE)
      );
  }

  private static void checkSSLContextSymbol(SubscriptionContext ctx, SymbolV2 contextSymbol, Tree locationForIssue) {
    boolean isUnsafeContext = isUnsafeDefaultContext(ctx, contextSymbol);
    Optional<AssignmentStatement> unsafeMaximumVersionStatement = findUnsafeMaximumVersionStatement(contextSymbol);
    if (isUnsafeContext && unsafeMaximumVersionStatement.isPresent()) {
      // Add the unsafe maximum version as a secondary location to the issue on the original location
      PreciseIssue issue = ctx.addIssue(locationForIssue, WEAK_PROTOCOL_MESSAGE);
      issue.secondary(unsafeMaximumVersionStatement.get(), "Unsafe maximum version specified here");
    } else if (unsafeMaximumVersionStatement.isPresent()) {
      // Create a standalone issue on the unsafe statement
      ctx.addIssue(unsafeMaximumVersionStatement.get(), WEAK_PROTOCOL_MESSAGE);
    } else if (isUnsafeContext) {
      // Create a standalone issue on the main location
      ctx.addIssue(locationForIssue, WEAK_PROTOCOL_MESSAGE);
    }
  }

  private static void checkOpenSSLContextSymbol(SubscriptionContext ctx, SymbolV2 contextSymbol, Tree locationForIssue) {
    if (!isSecurelyConfiguredOpenSSLContext(contextSymbol)) {
      ctx.addIssue(locationForIssue, WEAK_PROTOCOL_MESSAGE);
    }
  }

  private static boolean isUnsafeDefaultContext(SubscriptionContext ctx, SymbolV2 contextSymbol) {
    boolean isAllPython310OrAbove = PythonVersionUtils.areSourcePythonVersionsGreaterOrEqualThan(ctx.sourcePythonVersions(), PythonVersionUtils.Version.V_310);
    return !isAllPython310OrAbove && !isSecurelyConfigured(contextSymbol);
  }

  private static Optional<SymbolV2> getContextSymbol(Tree tree) {
    return Optional.ofNullable(TreeUtils.firstAncestorOfKind(tree, Tree.Kind.ASSIGNMENT_STMT))
      .map(AssignmentStatement.class::cast)
      .map(as -> as.lhsExpressions().get(0).expressions().get(0))
      .filter(Name.class::isInstance)
      .map(Name.class::cast)
      .map(Name::symbolV2);
  }

  private static boolean isSecurelyConfigured(SymbolV2 symbolV2) {
    // Check if secure through minimum_version or through security flags
    Set<String> securityFlags = collectSecurityFlags(symbolV2, "options");
    return symbolV2.usages().stream()
      .anyMatch(u -> isSettingSafeMinimumVersion(u.tree())) || securityFlags.containsAll(REQUIRED_SECURITY_FLAGS);
  }

  private static boolean isSecurelyConfiguredOpenSSLContext(SymbolV2 symbolV2) {
    // Check if secure through set_min_proto_version or through set_options
    return isSecureThroughMinProtoVersion(symbolV2) || isSecureThroughSetOptions(symbolV2);
  }

  private static boolean isSecureThroughMinProtoVersion(SymbolV2 symbolV2) {
    return symbolV2.usages().stream()
      .anyMatch(u -> {
        // Look for a call expression ancestor
        CallExpression callExpression = (CallExpression) TreeUtils.firstAncestorOfKind(u.tree(), Tree.Kind.CALL_EXPR);
        if (callExpression == null) {
          return false;
        }

        // Check if the callee type matches OpenSSL.SSL.Context.set_min_proto_version
        Symbol symbol = callExpression.calleeSymbol();
        if (symbol == null || !"set_min_proto_version".equals(symbol.name())) {
          return false;
        }

        // Check if any argument refers to TLS1_2_VERSION or TLS1_3_VERSION
        return callExpression.arguments().stream()
          .filter(RegularArgument.class::isInstance)
          .map(RegularArgument.class::cast)
          .map(RegularArgument::expression)
          .filter(HasSymbol.class::isInstance)
          .map(HasSymbol.class::cast)
          .map(HasSymbol::symbol)
          .filter(Objects::nonNull)
          .map(Symbol::fullyQualifiedName)
          .filter(Objects::nonNull)
          .anyMatch(fqn -> fqn.contains("TLS1_2_VERSION") || fqn.contains("TLS1_3_VERSION"));
      });
  }

  private static boolean isSecureThroughSetOptions(SymbolV2 symbolV2) {
    Set<String> securityFlags = collectOpenSSLSecurityFlags(symbolV2);
    return securityFlags.containsAll(REQUIRED_SECURITY_FLAGS);
  }

  private static Set<String> collectOpenSSLSecurityFlags(SymbolV2 symbolV2) {
    Set<String> securityFlags = new HashSet<>();
    symbolV2.usages().stream()
      .map(UsageV2::tree)
      .map(Tree::parent)
      .filter(QualifiedExpression.class::isInstance)
      .map(QualifiedExpression.class::cast)
      .filter(qe -> "set_options".equals(qe.name().name()))
      .map(Tree::parent)
      .filter(CallExpression.class::isInstance)
      .map(CallExpression.class::cast)
      .forEach(call -> {
        if (!call.arguments().isEmpty()) {
          TreeUtils.nthArgumentOrKeywordOptional(0, "options", call.arguments())
            .map(RegularArgument::expression)
            .ifPresent(expression -> collectSecurityFlagsFromExpression(expression, securityFlags));
        }
      });
    return securityFlags;
  }

  private static Set<String> collectSecurityFlags(SymbolV2 symbolV2, String propertyName) {
    Set<String> securityFlags = new HashSet<>();
    symbolV2.usages()
      .stream()
      .map(UsageV2::tree)
      .map(Tree::parent)
      .filter(QualifiedExpression.class::isInstance)
      .map(QualifiedExpression.class::cast)
      .filter(qe -> propertyName.equals(qe.name().name()))
      .map(qe -> TreeUtils.firstAncestorOfKind(qe, Tree.Kind.COMPOUND_ASSIGNMENT))
      .filter(CompoundAssignmentStatement.class::isInstance)
      .map(CompoundAssignmentStatement.class::cast)
      .map(CompoundAssignmentStatement::rhsExpression)
      .forEach(rhs -> collectSecurityFlagsFromExpression(rhs, securityFlags));
    return securityFlags;
  }

  private static void collectSecurityFlagsFromExpression(Expression expression, Set<String> securityFlags) {
    expression = Expressions.removeParentheses(expression);
    if (expression instanceof HasSymbol hasSymbol) {
      Optional.ofNullable(hasSymbol.symbol())
        .map(Symbol::fullyQualifiedName)
        .ifPresent(fqn ->
          REQUIRED_SECURITY_FLAGS.stream()
            .filter(fqn::contains)
            .forEach(securityFlags::add)
        );
    } else if (expression instanceof BinaryExpression binaryExpression) {
      // For binary expressions like a | b | c, process both sides
      collectSecurityFlagsFromExpression(binaryExpression.leftOperand(), securityFlags);
      collectSecurityFlagsFromExpression(binaryExpression.rightOperand(), securityFlags);
    }
    // Other expression types aren't relevant for our security flags
  }

  private static boolean isSettingSafeMinimumVersion(Tree tree) {
    return findVersionStatement(tree, "minimum_version", WeakSSLProtocolCheck::containsSafeVersion)
      .isPresent();
  }

  private static Optional<AssignmentStatement> findUnsafeMaximumVersionStatement(SymbolV2 symbolV2) {
    return symbolV2.usages().stream()
      .map(u -> findVersionStatement(u.tree(), "maximum_version", fqn -> !containsSafeVersion(fqn)))
      .filter(Optional::isPresent)
      .map(Optional::get)
      .findFirst();
  }

  private static boolean containsSafeVersion(String fullyQualifiedName) {
    return SAFE_VERSION_NAMES.stream().anyMatch(fullyQualifiedName::contains);
  }

  private static Optional<AssignmentStatement> findVersionStatement(Tree tree, String versionProperty, Predicate<String> versionPredicate) {
    return Optional.ofNullable(TreeUtils.firstAncestorOfKind(tree, Tree.Kind.ASSIGNMENT_STMT))
      .map(AssignmentStatement.class::cast)
      .filter(a -> isSettingVersionProperty(a, versionProperty))
      .filter(a -> Optional.of(a.assignedValue())
        .filter(HasSymbol.class::isInstance)
        .map(HasSymbol.class::cast)
        .map(HasSymbol::symbol)
        .map(Symbol::fullyQualifiedName)
        .filter(versionPredicate)
        .isPresent());
  }

  private static boolean isSettingVersionProperty(AssignmentStatement assignment, String versionProperty) {
    return assignment.lhsExpressions().stream()
      .flatMap(lhsExpr -> lhsExpr.expressions().stream())
      .filter(expr -> expr.is(Tree.Kind.QUALIFIED_EXPR))
      .map(QualifiedExpression.class::cast)
      .anyMatch(qexpr -> versionProperty.equals(qexpr.name().name()));
  }
}
