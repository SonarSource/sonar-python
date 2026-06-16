/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
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

import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Stream;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.ArgList;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringElement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.checks.cdk.ClearTextProtocolsCheckPart;
import org.sonar.python.tree.TreeUtils;
import org.sonarsource.analyzer.commons.appsec.CleartextProtocolFilter;

@Rule(key = "S5332")
public class ClearTextProtocolsCheck extends PythonSubscriptionCheck {
  private static final Set<String> CLEARTEXT_PROTOCOLS = CleartextProtocolFilter.getCleartextProtocols();
  private static final String SENSITIVE_HTTP_SERVER_START_FQN = "socketserver.BaseServer.serve_forever";
  private static final String SENSITIVE_HTTP_SERVER_BIND_FQN = "socketserver.BaseServer.server_bind";
  private static final Set<String> SENSITIVE_HTTP_SERVER_METHOD_NAMES = Set.of("serve_forever", "server_bind");
  private static final Set<String> SENSITIVE_HTTP_SERVER_CLASSES = Set.of("http.server.HTTPServer", "http.server.ThreadingHTTPServer");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.STRING_ELEMENT, ctx -> {
      Tree node = ctx.syntaxNode();
      String value = Expressions.unescape((StringElement) node);
      unsafeProtocol(value)
        // cleanup slashes
        .map(protocol -> protocol.substring(0, protocol.length() - 3))
        .ifPresent(protocol -> ctx.addIssue(node, message(protocol)));
    });
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> {
      CallExpression callExpression = (CallExpression) ctx.syntaxNode();
      Optional.ofNullable(callExpression.calleeSymbol())
        .map(Symbol::fullyQualifiedName)
        .flatMap(ClearTextProtocolsCheck::isUnsafeLib)
        .ifPresent(protocol -> ctx.addIssue(callExpression, message(protocol)));
    });

    context.registerSyntaxNodeConsumer(Tree.Kind.ASSIGNMENT_STMT, ctx -> handleAssignmentStatement((AssignmentStatement) ctx.syntaxNode(), ctx));

    context.registerSyntaxNodeConsumer(Tree.Kind.QUALIFIED_EXPR, ClearTextProtocolsCheck::checkServerCallFromSuper);

    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ClearTextProtocolsCheck::checkServerBindCalls);

    new ClearTextProtocolsCheckPart().initialize(context);
  }

  private static void checkServerCallFromSuper(SubscriptionContext ctx) {
    QualifiedExpression qualifiedExpression = (QualifiedExpression) ctx.syntaxNode();
    Optional.of(qualifiedExpression)
      .filter(qe -> SENSITIVE_HTTP_SERVER_METHOD_NAMES.contains(qe.name().name()))
      .filter(ClearTextProtocolsCheck::isCallToSensitiveSuperClass)
      .map(qe -> TreeUtils.firstAncestorOfKind(qe, Tree.Kind.CALL_EXPR))
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(CallExpression.class))
      .ifPresent(ce -> ctx.addIssue(ce, message("http")));
  }

  private static void checkServerBindCalls(SubscriptionContext ctx) {
    CallExpression callExpression = (CallExpression) ctx.syntaxNode();
    Optional.ofNullable(callExpression.calleeSymbol())
      .map(Symbol::fullyQualifiedName)
      .filter(SENSITIVE_HTTP_SERVER_BIND_FQN::equals)
      .filter(fqn -> isParentClassExtendingSensitiveClass(callExpression))
      .ifPresent(fqn -> ctx.addIssue(callExpression, message("http")));
  }

  private static boolean isCallToSensitiveSuperClass(QualifiedExpression expression) {
    return Optional.of(expression.qualifier())
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(CallExpression.class))
      .map(CallExpression::callee)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(Name.class))
      .map(Name::name)
      .filter("super"::equals)
      .filter(name -> isParentClassExtendingSensitiveClass(expression))
      .isPresent();
  }

  private static boolean isParentClassExtendingSensitiveClass(Tree expression) {
    return Optional.ofNullable(TreeUtils.firstAncestorOfKind(expression, Tree.Kind.CLASSDEF))
      .map(ClassDef.class::cast)
      .map(ClassDef::args)
      .map(ArgList::arguments)
      .map(ClearTextProtocolsCheck::getClassFQNFromArgument)
      .map(arguments -> arguments.anyMatch(SENSITIVE_HTTP_SERVER_CLASSES::contains))
      .orElse(false);
  }

  private static Stream<String> getClassFQNFromArgument(List<Argument> arguments) {
    return arguments.stream()
      .map(TreeUtils.toInstanceOfMapper(RegularArgument.class))
      .filter(Objects::nonNull)
      .map(RegularArgument::expression)
      .filter(HasSymbol.class::isInstance)
      .map(HasSymbol.class::cast)
      .map(HasSymbol::symbol)
      .filter(Objects::nonNull)
      .map(Symbol::fullyQualifiedName);
  }

  private static void handleAssignmentStatement(AssignmentStatement assignmentStatement, SubscriptionContext ctx) {
    if (assignmentStatement.lhsExpressions().size() > 1) {
      // avoid potential FPs
      return;
    }
    Expression lhs = assignmentStatement.lhsExpressions().get(0).expressions().get(0);
    if (lhs instanceof HasSymbol hasSymbol) {
      Symbol symbol = hasSymbol.symbol();
      if (symbol == null) {
        return;
      }
      if (lhs.type().canOnlyBe("smtplib.SMTP")) {
        boolean usesEncryption = symbol.usages().stream().anyMatch(u -> {
          Tree tree = TreeUtils.firstAncestorOfKind(u.tree(), Tree.Kind.CALL_EXPR);
          if (tree != null) {
            Symbol calleeSymbol = ((CallExpression) tree).calleeSymbol();
            return calleeSymbol != null && "smtplib.SMTP.starttls".equals(calleeSymbol.fullyQualifiedName());
          }
          return false;
        });
        if (!usesEncryption) {
          ctx.addIssue(assignmentStatement.assignedValue(), "Make sure STARTTLS is used to upgrade to a secure connection using SSL/TLS.");
        }
      }
    }
  }

  private static Optional<String> unsafeProtocol(String literalValue) {
    return CLEARTEXT_PROTOCOLS.stream()
      .filter(literalValue::startsWith)
      .filter(p -> {
        String rest = literalValue.substring(p.length());
        if (rest.isEmpty()) {
          // Bare scheme string (e.g. "http://") — always flag
          return true;
        }
        char first = rest.charAt(0);
        if (first == '/' || first == '?' || first == '#') {
          // No authority component (e.g. "http:///path") — no host to evaluate
          return false;
        }
        return !CleartextProtocolFilter.isSafeWithoutTls(literalValue);
      })
      .findFirst();
  }

  private static Optional<String> isUnsafeLib(String qualifiedName) {
    if ("telnetlib.Telnet".equals(qualifiedName)) {
      return Optional.of("telnet");
    }
    if ("ftplib.FTP".equals(qualifiedName)) {
      return Optional.of("ftp");
    }
    if (SENSITIVE_HTTP_SERVER_START_FQN.equals(qualifiedName)) {
      return Optional.of("http");
    }
    return Optional.empty();
  }

  private static String message(String protocol) {
    return CleartextProtocolFilter.getIssueMessage(protocol)
      .orElse("Using " + protocol + " protocol is insecure. Use a secure alternative instead.");
  }
}
