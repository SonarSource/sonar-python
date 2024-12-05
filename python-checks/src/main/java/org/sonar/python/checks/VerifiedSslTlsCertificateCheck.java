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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.function.Predicate;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.DictionaryLiteral;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.KeyValuePair;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.NumericLiteral;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.UnpackingExpression;
import org.sonar.plugins.python.api.tree.WithItem;
import org.sonar.plugins.python.api.tree.WithStatement;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.RegularArgumentImpl;
import org.sonar.python.tree.TreeUtils;

import static java.util.Optional.ofNullable;

// https://jira.sonarsource.com/browse/RSPEC-4830
@Rule(key = "S4830")
public class VerifiedSslTlsCertificateCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Enable server certificate validation on this SSL/TLS connection.";
  private static final String VERIFY_NONE = Fqn.ssl("VERIFY_NONE");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.WITH_STMT, VerifiedSslTlsCertificateCheck::verifyAioHttpWithSession);
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, VerifiedSslTlsCertificateCheck::sslSetVerifyCheck);
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, VerifiedSslTlsCertificateCheck::requestsCheck);
    context.registerSyntaxNodeConsumer(Tree.Kind.REGULAR_ARGUMENT, VerifiedSslTlsCertificateCheck::standardSslCheckForRegularArgument);
    context.registerSyntaxNodeConsumer(Tree.Kind.ASSIGNMENT_STMT, VerifiedSslTlsCertificateCheck::standardSslCheckForAssignmentStatement);
  }

  private static void verifyAioHttpWithSession(SubscriptionContext ctx) {
    var withStatement = (WithStatement) ctx.syntaxNode();
    withStatement.withItems()
      .stream()
      .filter(item -> Optional.of(item)
        .map(WithItem::test)
        .flatMap(TreeUtils.toOptionalInstanceOfMapper(CallExpression.class))
        .map(CallExpression::calleeSymbol)
        .map(Symbol::fullyQualifiedName)
        .filter("aiohttp.ClientSession"::equals)
        .isPresent())
      .map(WithItem::expression)
      .map(TreeUtils.toOptionalInstanceOfMapper(Name.class))
      .filter(Optional::isPresent)
      .map(Optional::get)
      .map(HasSymbol::symbol)
      .filter(Objects::nonNull)
      .forEach(symbol -> verifyAioHttpSessionSymbolUsages(ctx, symbol));
  }

  private static void verifyAioHttpSessionSymbolUsages(SubscriptionContext ctx, Symbol sessionSymbol) {
    sessionSymbol.usages()
      .stream()
      .filter(usage -> usage.kind() == Usage.Kind.OTHER)
      .map(Usage::tree)
      .map(t -> TreeUtils.firstAncestorOfKind(t, Tree.Kind.CALL_EXPR))
      .map(TreeUtils.toOptionalInstanceOfMapper(CallExpression.class))
      .filter(Optional::isPresent)
      .map(Optional::get)
      .forEach(sessionCallExpr -> verifyVulnerableMethods(ctx, sessionCallExpr, VERIFY_SSL_ARG_NAMES));
  }

  /** Fully qualified name of the <code>set_verify</code> used in <code>sslSetVerifyCheck</code>. */
  private static final String SET_VERIFY = Fqn.context("set_verify");

  /**
   * Check for the <code>OpenSSL.SSL.Context.set_verify</code> flag settings.
   *
   * Searches for `set_verify` invocations on instances of `OpenSSL.SSL.Context`,
   * extracts the flags from the first argument, checks that the combination of flags is secure.
   *
   * @param subscriptionContext the subscription context passed by <code>Context.registerSyntaxNodeConsumer</code>.
   */
  private static void sslSetVerifyCheck(SubscriptionContext subscriptionContext) {

    CallExpression callExpr = (CallExpression) subscriptionContext.syntaxNode();

    boolean isSetVerifyInvocation = ofNullable(callExpr.calleeSymbol())
      .map(Symbol::fullyQualifiedName)
      .filter(SET_VERIFY::equals)
      .isPresent();

    if (isSetVerifyInvocation) {
      List<Argument> args = callExpr.arguments();
      if (!args.isEmpty()) {
        Tree flagsArgument = args.get(0);
        if (flagsArgument.is(Tree.Kind.REGULAR_ARGUMENT)) {
          Set<QualifiedExpression> flags = extractFlags(((RegularArgumentImpl) flagsArgument).expression());
          checkFlagSettings(flags).ifPresent(issue -> subscriptionContext.addIssue(issue.token, MESSAGE));
        }
      }
    }
  }

  /** Helper methods for generating FQNs frequently used in this check. */
  private static class Fqn {
    private static String context(@SuppressWarnings("SameParameterValue") String method) {
      return ssl("Context." + method);
    }

    private static String ssl(String property) {
      return "OpenSSL.SSL." + property;
    }
  }

  /**
   * Recursively deconstructs binary trees of expressions separated with `|`-ors,
   * and collects the leafs that look like qualified expressions.
   */
  private static HashSet<QualifiedExpression> extractFlags(Tree flagsSubexpr) {
    if (flagsSubexpr.is(Tree.Kind.QUALIFIED_EXPR)) {
      // Base case: e.g. `SSL.VERIFY_NONE`
      return new HashSet<>(Collections.singletonList((QualifiedExpression) flagsSubexpr));
    } else if (flagsSubexpr.is(Tree.Kind.BITWISE_OR)) {
      // recurse into left and right branch
      BinaryExpression orExpr = (BinaryExpression) flagsSubexpr;
      HashSet<QualifiedExpression> flags = extractFlags(orExpr.leftOperand());
      flags.addAll(extractFlags(orExpr.rightOperand()));
      return flags;
    } else {
      // failed to interpret. Ignore leaf.
      return new HashSet<>();
    }
  }

  /**
   * Checks whether a combination of flags is valid,
   * optionally returns a message and a token if there is something wrong.
   */
  private static Optional<IssueReport> checkFlagSettings(Set<QualifiedExpression> flags) {
    for (QualifiedExpression qe : flags) {
      Symbol symb = qe.symbol();
      if (symb != null) {
        String fqn = symb.fullyQualifiedName();
        if (VERIFY_NONE.equals(fqn)) {
          return Optional.of(new IssueReport(
            "Omitting the check of the peer certificate is dangerous.",
            qe.lastToken()));
        }
      }
    }
    return Optional.empty();
  }

  /** Message and a token closest to the problematic position. Glorified <code>Pair&lt;A,B&gt;</code>. */
  private static class IssueReport {
    final String message;
    final Token token;

    private IssueReport(String message, Token token) {
      this.message = message;
      this.token = token;
    }
  }

  public static final Set<String> VERIFY_ARG_NAME = Set.of("verify");
  public static final Set<String> VERIFY_SSL_ARG_NAMES = Set.of("verify_ssl", "ssl");
  /**
   * Set of FQNs of methods in <code>requests</code>-module that have the vulnerable <code>verify</code>-option.
   */
  private static final Set<String> CALLS_WHERE_TO_ENFORCE_TRUE_ARGUMENT = Set.of(
    "requests.api.request",
    "requests.api.get",
    "requests.api.head",
    "requests.api.post",
    "requests.api.put",
    "requests.api.delete",
    "requests.api.patch",
    "requests.api.options",
    "httpx.request",
    "httpx.stream",
    "httpx.get",
    "httpx.options",
    "httpx.head",
    "httpx.post",
    "httpx.put",
    "httpx.patch",
    "httpx.delete",
    "httpx.Client",
    "httpx.AsyncClient");

  private static void requestsCheck(SubscriptionContext subscriptionContext) {
    var callExpr = (CallExpression) subscriptionContext.syntaxNode();
    var isVulnerableMethod = ofNullable(callExpr.calleeSymbol())
      .map(Symbol::fullyQualifiedName)
      .filter(CALLS_WHERE_TO_ENFORCE_TRUE_ARGUMENT::contains)
      .isPresent();

    if (isVulnerableMethod) {
      verifyVulnerableMethods(subscriptionContext, callExpr, VERIFY_ARG_NAME);
    }
  }

  private static void verifyVulnerableMethods(SubscriptionContext ctx, CallExpression callExpr, Set<String> argumentNames) {
    var verifyRhs = searchVerifyAssignment(callExpr, argumentNames)
      .or(() -> searchVerifyInKwargs(callExpr, argumentNames));

    verifyRhs.ifPresent(sensitiveSettingExpressions -> sensitiveSettingExpressions
      .stream()
      .filter(rhs -> Expressions.isFalsy(rhs) || isFalsyCollection(rhs))
      .findFirst()
      .ifPresent(rhs -> addIssue(ctx, sensitiveSettingExpressions, rhs))
    );
  }

  private static void addIssue(SubscriptionContext ctx, List<Expression> sensitiveSettingExpressions, Expression rhs) {
    var issue = ctx.addIssue(rhs, MESSAGE);
    // report everything except the last one as secondary locations.
    sensitiveSettingExpressions.stream()
      .filter(v -> v != rhs)
      .forEach(v -> issue.secondary(v, "Dictionary is passed here as **kwargs."));
  }

  /**
   * Attempts to find the expression in <code>verify = expr</code> explicitly keyworded parameter assignment.
   *
   * @return The <code>expr</code> part on the right hand side of the assignment.
   */
  private static Optional<List<Expression>> searchVerifyAssignment(CallExpression callExpr, Set<String> argumentNames) {
    var args = callExpr.arguments()
      .stream()
      .filter(RegularArgument.class::isInstance)
      .map(RegularArgument.class::cast)
      .filter(regArg -> Optional.of(regArg)
        .map(RegularArgument::keywordArgument)
        .map(Name::name)
        .filter(argumentNames::contains)
        .isPresent())
      .map(RegularArgument::expression)
      .toList();
    return Optional.of(args).filter(Predicate.not(List::isEmpty));
  }

  /**
   * Attempts to find the <code>rhs</code> in some definition <code>kwargs = { 'verify': rhs }</code>
   * of <code>kwargs</code> used in the arguments of the given <code>callExpression</code>.
   *
   * Returns list of problematic expressions in the reverse order of importance (the <code>kwargs</code>-argument comes
   * first, the setting in the dictionary comes last).
   */
  private static Optional<List<Expression>> searchVerifyInKwargs(CallExpression callExpression, Set<String> argumentNames) {
    // Finds first unpacking argument (**kwargs),
    // attempts to find the definition with the dictionary,
    // then attempts to find a bad setting in the dictionary,
    // and finally returns the list with both the `kwargs`-argument and the bad setting in the dictionary.
    return callExpression.arguments().stream()
      .filter(UnpackingExpression.class::isInstance)
      .map(arg -> ((UnpackingExpression) arg).expression())
      .filter(Name.class::isInstance)
      .findFirst()
      .flatMap(name -> Optional.ofNullable(Expressions.singleAssignedValue((Name) name))
        .filter(DictionaryLiteral.class::isInstance)
        .flatMap(dict -> searchDangerousVerifySettingInDictionary((DictionaryLiteral) dict, argumentNames)
          .map(settingInDict -> Arrays.asList(name, settingInDict))));
  }

  /** Searches for a dangerous falsy <code>verify: False</code> in a dictionary literal. */
  private static Optional<Expression> searchDangerousVerifySettingInDictionary(DictionaryLiteral dict, Set<String> argumentNames) {
    return dict.elements().stream()
      .filter(KeyValuePair.class::isInstance)
      .map(KeyValuePair.class::cast)
      .filter(kvp -> Optional.of(kvp.key())
        .filter(StringLiteral.class::isInstance)
        .map(StringLiteral.class::cast)
        .map(StringLiteral::trimmedQuotesValue)
        .filter(argumentNames::contains)
        .isPresent())
      .findFirst()
      .map(KeyValuePair::value);
  }

  /**
   * Checks whether an expression is obviously a falsy collection (e.g. <code>set()</code> or <code>range(0)</code>).
   */
  private static boolean isFalsyCollection(Expression expr) {
    if (expr instanceof CallExpression callExpr) {
      Optional<String> fqnOpt = Optional.ofNullable(callExpr.calleeSymbol()).map(Symbol::fullyQualifiedName);
      if (fqnOpt.isPresent()) {
        String fqn = fqnOpt.get();
        return isFalsyNoArgCollectionConstruction(callExpr, fqn) || isFalsyRange(callExpr, fqn);
      }
    }
    return false;
  }

  /** FQNs of collection constructors that yield a falsy collection if invoked without arguments. */
  private static final Set<String> NO_ARG_FALSY_COLLECTION_CONSTRUCTORS = new HashSet<>(Arrays.asList(
    "set", "list", "dict"));

  /** Detects expressions like <code>dict()</code> or <code>list()</code>. */
  private static boolean isFalsyNoArgCollectionConstruction(CallExpression callExpr, String fqn) {
    return NO_ARG_FALSY_COLLECTION_CONSTRUCTORS.contains(fqn) && callExpr.arguments().isEmpty();
  }

  private static boolean isFalsyRange(CallExpression callExpr, String fqn) {
    if ("range".equals(fqn) && callExpr.arguments().size() == 1) {
      // `range(0)` is also falsy
      Argument firstArg = callExpr.arguments().get(0);
      if (firstArg instanceof RegularArgument regArg) {
        Expression firstArgExpr = regArg.expression();
        if (firstArgExpr.is(Tree.Kind.NUMERIC_LITERAL)) {
          NumericLiteral num = (NumericLiteral) firstArgExpr;
          return num.valueAsLong() == 0L;
        }
      }
    }
    return false;
  }

  private static void standardSslCheckForAssignmentStatement(SubscriptionContext subscriptionContext) {
    AssignmentStatement asgnStmt = (AssignmentStatement) subscriptionContext.syntaxNode();

    Optional<VulnerabilityAndProblematicToken> vulnTokOpt = isVulnerableMethodCall(asgnStmt.assignedValue());
    vulnTokOpt.ifPresent(vulnTok -> asgnStmt
      .lhsExpressions()
      .stream()
      .flatMap(it -> it.expressions().stream())
      .findFirst()
      .filter(Name.class::isInstance)
      .map(expr -> ((Name) expr).symbol())
      .ifPresent(symb -> {
        for (Usage u : selectRelevantModifyingUsages(symb.usages(), vulnTok.token.line())) {
          searchForVerifyModeOverride(u).ifPresent(vulnTok::overrideBy);
        }
        if (vulnTok.isVulnerable) {
          subscriptionContext.addIssue(vulnTok.token, MESSAGE);
        }
      }));
  }

  private static void standardSslCheckForRegularArgument(SubscriptionContext subscriptionContext) {
    var argument = (RegularArgument) subscriptionContext.syntaxNode();
    isVulnerableMethodCall(argument.expression())
      .ifPresent(vulnTok -> subscriptionContext.addIssue(vulnTok.token, MESSAGE));
  }

  /** Finds the next higher line where a binding usage occurs. */
  private static int findNextAssignmentLine(List<Usage> usages, int firstAssignmentLine) {
    int closestHigher = Integer.MAX_VALUE;
    for (Usage u : usages) {
      if (u.isBindingUsage()) {
        int line = u.tree().firstToken().line();
        if (line > firstAssignmentLine && line <= closestHigher) {
          closestHigher = line;
        }
      }
    }
    return closestHigher;
  }

  /**
   * Selects all non-binding usages between first assignment and next assignment.
   *
   * We assume that in a vast majority of cases, there will be no complex control flow between the instantiation
   * of the context and the modification of the settings, thus selecting and sorting usages by line numbers
   * should suffice here.
   */
  private static List<Usage> selectRelevantModifyingUsages(List<Usage> usages, int firstAssignmentLine) {
    int nextAssignmentLine = findNextAssignmentLine(usages, firstAssignmentLine);
    ArrayList<Usage> result = new ArrayList<Usage>();
    usages.stream().filter(u -> {
      int line = u.tree().firstToken().line();
      return !u.isBindingUsage() && line > firstAssignmentLine && line < nextAssignmentLine;
    }).forEach(u -> result.add(u));
    result.sort(Comparator.comparing(u -> u.tree().firstToken().line()));
    return result;
  }

  /**
   * Map from FQNs of sensitive context factories to the boolean that determines whether default settings are dangerous.
   */
  private static final Map<String, Boolean> VULNERABLE_CONTEXT_FACTORIES = Map.of(
    "ssl._create_unverified_context", true,
    "ssl._create_stdlib_context", true,
    "ssl.create_default_context", false,
    "ssl._create_default_https_context", false);

  /** Pair and a mutable cell for combining all updates to <code>verify_mode</code>. */
  private static class VulnerabilityAndProblematicToken {
    boolean isInvisibleDefaultPreset;
    boolean isVulnerable;
    Token token;

    VulnerabilityAndProblematicToken(
      boolean isVulnerable,
      Token token,
      boolean isInvisibleDefaultPreset) {
      this.isVulnerable = isVulnerable;
      this.token = token;
      this.isInvisibleDefaultPreset = isInvisibleDefaultPreset;
    }

    void overrideBy(VulnerabilityAndProblematicToken overridingAssignment) {
      this.isInvisibleDefaultPreset = false;
      this.isVulnerable = overridingAssignment.isVulnerable;
      this.token = overridingAssignment.token;
    }
  }

  /**
   * Searches an expression for a factory invocation of shape <code>ssl.somehow_create_context</code>,
   * if found, returns the token of the callee, together with the boolean that indicates whether the default settings
   * are dangerous.
   */
  private static Optional<VulnerabilityAndProblematicToken> isVulnerableMethodCall(Expression expr) {
    if (expr instanceof CallExpression callExpression) {
      Symbol calleeSymbol = callExpression.calleeSymbol();
      if (calleeSymbol != null) {
        String fqn = calleeSymbol.fullyQualifiedName();
        if (fqn != null && VULNERABLE_CONTEXT_FACTORIES.containsKey(fqn)) {
          boolean isVulnerable = VULNERABLE_CONTEXT_FACTORIES.get(fqn);
          return Optional.of(new VulnerabilityAndProblematicToken(
            isVulnerable,
            callExpression.callee().lastToken(),
            true));
        }
      }
    }
    return Optional.empty();
  }

  private static Optional<VulnerabilityAndProblematicToken> searchForVerifyModeOverride(Usage u) {
    if (!u.isBindingUsage()) {
      return Optional.of(u)
        .map(Usage::tree)
        .map(Tree::parent)
        .filter(QualifiedExpression.class::isInstance)
        .map(QualifiedExpression.class::cast)
        .filter(qe -> "verify_mode".equals(qe.name().name()))
        .map(QualifiedExpression::parent)
        .map(Tree::parent)
        .filter(AssignmentStatement.class::isInstance)
        .map(ae -> ((AssignmentStatement) ae).assignedValue())
        .filter(QualifiedExpression.class::isInstance)
        .flatMap(qe -> Optional
          .ofNullable(((QualifiedExpression) qe).symbol())
          .map(symb -> new VulnerabilityAndProblematicToken(
            "ssl.CERT_NONE".equals(symb.fullyQualifiedName()),
            qe.lastToken(),
            false)));
    }
    return Optional.empty();
  }
}
