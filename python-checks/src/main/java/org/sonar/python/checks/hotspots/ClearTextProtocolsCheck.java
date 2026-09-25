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
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.v2.SymbolV2;
import org.sonar.plugins.python.api.symbols.v2.UsageV2;
import org.sonar.plugins.python.api.tree.ArgList;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringElement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
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
  private static final TypeMatcher STR_METHOD_MATCHER = TypeMatchers.isFunctionOwnerSatisfying(TypeMatchers.isOrExtendsType("builtins.str"));
  private static final TypeMatcher STR_TYPE_MATCHER = TypeMatchers.isType("builtins.str");
  // Receiver-side check (not isFunctionOwnerSatisfying: serve_forever's *declaring* class is always
  // socketserver.BaseServer, since it's never overridden - that's the FP this matcher fixes).
  // isObjectInstanceOf covers an instance receiver (server.serve_forever()); isOrExtendsType covers
  // the class itself used as the receiver in an unbound-method call (HTTPServer.serve_forever(self)).
  private static final TypeMatcher HTTP_SERVER_INSTANCE_MATCHER = TypeMatchers.any(
    SENSITIVE_HTTP_SERVER_CLASSES.stream().flatMap(fqn -> Stream.of(TypeMatchers.isObjectInstanceOf(fqn), TypeMatchers.isOrExtendsType(fqn))).toList());
  private static final TypeMatcher SSL_CALL_MATCHER = TypeMatchers.withFQNPrefix("ssl.");
  // Methods where the first (non-self) argument is a scheme/prefix used only for identification, not sent over the network
  private static final Set<String> PROTOCOL_IDENTIFICATION_METHODS = Set.of("startswith", "replace", "removeprefix", "removesuffix");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.STRING_ELEMENT, ctx -> {
      Tree node = ctx.syntaxNode();
      String value = Expressions.unescape((StringElement) node);
      unsafeProtocol(value)
        .filter(protocol -> !isProtocolIdentificationArgument(node, ctx))
        // cleanup slashes
        .map(protocol -> protocol.substring(0, protocol.length() - 3))
        .ifPresent(protocol -> ctx.addIssue(node, message(protocol)));
    });
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> {
      CallExpression callExpression = (CallExpression) ctx.syntaxNode();
      Optional.ofNullable(callExpression.calleeSymbol())
        .map(Symbol::fullyQualifiedName)
        .flatMap(ClearTextProtocolsCheck::isUnsafeLib)
        .filter(protocol -> !"http".equals(protocol) || isSensitiveUnprotectedHttpServerCall(callExpression, ctx))
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

  /**
   * Gates the "http" finding for a direct {@code serve_forever()} call. Two checks not implied by
   * the FQN match alone (serve_forever is inherited, never overridden, from socketserver.BaseServer):
   * <ul>
   *   <li>the receiver must actually be an HTTPServer/ThreadingHTTPServer instance, not merely
   *   something that happens to inherit serve_forever from the same base class. For an unbound call
   *   ({@code Cls.serve_forever(self)}), the object that actually serves is the argument, not the
   *   qualifier {@code Cls} - which may be any ancestor in the MRO (e.g.
   *   {@code socketserver.TCPServer.serve_forever(self)} on a real HTTPServer subclass). self's type
   *   isn't reliably resolvable here, so that case falls back to the same structural
   *   enclosing-class-bases check already used by {@link #checkServerBindCalls} and
   *   {@link #checkServerCallFromSuper} instead of inferring the argument's type;</li>
   *   <li>there must be no evidence that the socket was TLS-wrapped in the same scope as the
   *   server's construction (same permissive, dataflow-free style as the SMTP STARTTLS check above:
   *   any ssl.* call anywhere in scope suppresses, whether or not it actually secures this server,
   *   and regardless of call order).</li>
   * </ul>
   * Known, accepted limitations: no real dataflow to the specific socket; doesn't cover
   * construction and TLS setup happening in different methods (e.g. {@code __init__} vs
   * {@code start()}); no ordering requirement between the wrap and {@code serve_forever()}.
   */
  private static boolean isSensitiveUnprotectedHttpServerCall(CallExpression callExpression, SubscriptionContext ctx) {
    if (!(callExpression.callee() instanceof QualifiedExpression qualifiedExpression)) {
      return true;
    }
    Expression receiver = qualifiedExpression.qualifier();
    if (HTTP_SERVER_INSTANCE_MATCHER.evaluateFor(receiver, ctx) == TriBool.FALSE
      && !isParentClassExtendingSensitiveClass(callExpression)) {
      return false;
    }
    return !hasTlsEvidenceInScope(callExpression, receiver, ctx);
  }

  private static boolean hasTlsEvidenceInScope(CallExpression serveForeverCall, Expression receiver, SubscriptionContext ctx) {
    Tree anchor = findConstructorCall(receiver).orElse(serveForeverCall);
    // enclosingScope always finds either the FunctionDef body or the FileInput root - a call
    // expression is always part of a parsed file, so it never has no scope at all.
    Tree scope = enclosingScope(anchor);
    SslCallDetector detector = new SslCallDetector(ctx);
    scope.accept(detector);
    return detector.found;
  }

  private static Optional<Tree> findConstructorCall(Expression receiver) {
    if (receiver instanceof CallExpression constructorCall) {
      return Optional.of(constructorCall);
    }
    if (receiver instanceof Name name) {
      return Optional.ofNullable(name.symbolV2())
        .flatMap(SymbolV2::getSingleBindingUsage)
        .map(UsageV2::tree)
        .map(bindingTree -> TreeUtils.firstAncestorOfKind(bindingTree, Tree.Kind.ASSIGNMENT_STMT))
        .map(AssignmentStatement.class::cast)
        .map(AssignmentStatement::assignedValue);
    }
    return Optional.empty();
  }

  private static Tree enclosingScope(Tree node) {
    Tree functionDef = TreeUtils.firstAncestorOfKind(node, Tree.Kind.FUNCDEF);
    if (functionDef != null) {
      return ((FunctionDef) functionDef).body();
    }
    return TreeUtils.firstAncestorOfClass(node, FileInput.class);
  }

  private static class SslCallDetector extends BaseTreeVisitor {
    private final SubscriptionContext ctx;
    private boolean found = false;

    private SslCallDetector(SubscriptionContext ctx) {
      this.ctx = ctx;
    }

    @Override
    public void visitCallExpression(CallExpression callExpression) {
      if (SSL_CALL_MATCHER.evaluateFor(callExpression.callee(), ctx) == TriBool.TRUE) {
        found = true;
      }
      super.visitCallExpression(callExpression);
    }

    @Override
    public void visitFunctionDef(FunctionDef functionDef) {
      // Nested function bodies are a different scope; they aren't executed just by being defined.
    }
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

  private static boolean isProtocolIdentificationArgument(Tree stringElement, SubscriptionContext context) {
    Tree argumentParent = stringElement.parent().parent();
    while (argumentParent.is(Tree.Kind.TUPLE, Tree.Kind.PARENTHESIZED)) {
      argumentParent = argumentParent.parent();
    }
    if (!(argumentParent instanceof RegularArgument argument) || argument.keywordArgument() != null) {
      return false;
    }
    Tree callTree = TreeUtils.firstAncestorOfKind(argument, Tree.Kind.CALL_EXPR);
    // UNKNOWN owner type stays exempt (same tradeoff as SONARPY-4533); on an unresolved receiver this can hide a
    // real literal for a same-named non-str method (e.g. datetime.replace, dataclasses.replace, DataFrame.replace)
    if (!(callTree instanceof CallExpression callExpression)
      || !(callExpression.callee() instanceof QualifiedExpression callee)
      || !PROTOCOL_IDENTIFICATION_METHODS.contains(callee.name().name())
      || STR_METHOD_MATCHER.evaluateFor(callee, context) == TriBool.FALSE) {
      return false;
    }
    // unbound str.startswith(self, prefix) / str.replace(self, old, new) shifts the identification argument to the second position
    int identificationIndex = STR_TYPE_MATCHER.evaluateFor(callee.qualifier(), context) == TriBool.TRUE ? 1 : 0;
    List<Argument> arguments = callExpression.arguments();
    // only the first (scheme-identifying) argument is exempt — e.g. replace's replacement/new argument still gets flagged
    return arguments.size() > identificationIndex && arguments.get(identificationIndex) == argument;
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
