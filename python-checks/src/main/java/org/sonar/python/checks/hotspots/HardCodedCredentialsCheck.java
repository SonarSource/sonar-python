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

import java.net.MalformedURLException;
import java.net.URL;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import org.sonar.check.Rule;
import org.sonar.check.RuleProperty;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.DictionaryLiteral;
import org.sonar.plugins.python.api.tree.DictionaryLiteralElement;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionList;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.KeyValuePair;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Parameter;
import org.sonar.plugins.python.api.tree.ParameterList;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringElement;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.SubscriptionExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S2068")
public class HardCodedCredentialsCheck extends PythonSubscriptionCheck {

  private static final String DEFAULT_CREDENTIAL_WORDS = "password,passwd,pwd,passphrase";

  private static final String FLASK_CONFIG_ASSIGNMENT_FQN = "flask.app.Flask.config";
  private static final String FLASK_CONFIG_CREDENTIAL_KEY = "SECRET_KEY";

  @RuleProperty(
    key = "credentialWords",
    description = "Comma separated list of words identifying potential credentials",
    defaultValue = DEFAULT_CREDENTIAL_WORDS)
  public String credentialWords = DEFAULT_CREDENTIAL_WORDS;

  public static final String MESSAGE = "\"%s\" detected here, review this potentially hard-coded credential.";

  private List<Pattern> variablePatterns = null;
  private List<Pattern> literalPatterns = null;
  private Map<String, Integer> sensitiveArgumentByFQN;

  private Map<String, Integer> sensitiveArgumentByFQN() {
    if (sensitiveArgumentByFQN == null) {
      sensitiveArgumentByFQN = new HashMap<>();
      sensitiveArgumentByFQN.put("mysql.connector.connect", 2);
      sensitiveArgumentByFQN.put("mysql.connector.connection.MySQLConnection", 2);
      sensitiveArgumentByFQN.put("pymysql.connect", 2);
      sensitiveArgumentByFQN.put("pymysql.connections.connect", 2);
      sensitiveArgumentByFQN.put("pymysql.connections.Connection", 2);
      sensitiveArgumentByFQN.put("psycopg2.connect", 2);
      sensitiveArgumentByFQN.put("pgdb.connect", 2);
      sensitiveArgumentByFQN.put("pgdb.connect.connect", 2);
      sensitiveArgumentByFQN.put("pg.DB", 5);
      sensitiveArgumentByFQN.put("pg.connect", 5);
      sensitiveArgumentByFQN = Collections.unmodifiableMap(sensitiveArgumentByFQN);
    }
    return sensitiveArgumentByFQN;
  }

  private Stream<Pattern> variablePatterns() {
    if (variablePatterns == null) {
      variablePatterns = toPatterns("");
    }
    return variablePatterns.stream();
  }

  private Stream<Pattern> literalPatterns() {
    if (literalPatterns == null) {
      // Avoid raising on prepared statements
      String credentials = Stream.of(credentialWords.split(","))
        .map(String::trim).collect(Collectors.joining("|"));
      literalPatterns = toPatterns("=(?!.*(" + credentials + "))[^:%'?{\\s]+");
    }
    return literalPatterns.stream();
  }

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Kind.ASSIGNMENT_STMT, ctx -> handleAssignmentStatement((AssignmentStatement) ctx.syntaxNode(), ctx));
    context.registerSyntaxNodeConsumer(Kind.STRING_LITERAL, ctx -> handleStringLiteral((StringLiteral) ctx.syntaxNode(), ctx));
    context.registerSyntaxNodeConsumer(Kind.CALL_EXPR, ctx -> handleCallExpression((CallExpression) ctx.syntaxNode(), ctx));
    context.registerSyntaxNodeConsumer(Kind.REGULAR_ARGUMENT, ctx -> handleRegularArgument((RegularArgument) ctx.syntaxNode(), ctx));
    context.registerSyntaxNodeConsumer(Kind.PARAMETER_LIST, ctx -> handleParameterList((ParameterList) ctx.syntaxNode(), ctx));
    context.registerSyntaxNodeConsumer(Kind.DICTIONARY_LITERAL, ctx -> handleDictionaryLiteral((DictionaryLiteral) ctx.syntaxNode(), ctx));
  }

  private void handleDictionaryLiteral(DictionaryLiteral dictionaryLiteral, SubscriptionContext ctx) {
    for (DictionaryLiteralElement dictionaryLiteralElement : dictionaryLiteral.elements()) {
      if (dictionaryLiteralElement.is(Kind.KEY_VALUE_PAIR)) {
        KeyValuePair keyValuePair = (KeyValuePair) dictionaryLiteralElement;
        checkKeyValuePair(keyValuePair, ctx);
      }
    }
  }

  private void checkKeyValuePair(KeyValuePair keyValuePair, SubscriptionContext ctx) {
    if (keyValuePair.key().is(Kind.STRING_LITERAL) && keyValuePair.value().is(Kind.STRING_LITERAL)) {
      String matchedCredential = matchedCredential(((StringLiteral) keyValuePair.key()).trimmedQuotesValue(), variablePatterns());
      if (matchedCredential != null) {
        StringLiteral literal = (StringLiteral) keyValuePair.value();
        if (isSuspiciousStringLiteral(literal)) {
          ctx.addIssue(keyValuePair, String.format(MESSAGE, matchedCredential));
        }
      }
    }
  }

  private void handleParameterList(ParameterList parameterList, SubscriptionContext ctx) {
    for (Parameter parameter : parameterList.nonTuple()) {
      Name parameterName = parameter.name();
      if (parameterName == null) {
        continue;
      }
      Expression defaultValue = parameter.defaultValue();
      String matchedCredential = matchedCredential(parameterName.name(), variablePatterns());
      if (matchedCredential != null && defaultValue != null && isSuspiciousStringLiteral(defaultValue)) {
        ctx.addIssue(parameter, String.format(MESSAGE, matchedCredential));
      }
    }
  }

  private void handleRegularArgument(RegularArgument regularArgument, SubscriptionContext ctx) {
    Name keywordArgument = regularArgument.keywordArgument();
    if (keywordArgument != null) {
      String matchedCredential = matchedCredential(keywordArgument.name(), variablePatterns());
      if (matchedCredential != null && isSuspiciousStringLiteral(regularArgument.expression())) {
        ctx.addIssue(regularArgument, String.format(MESSAGE, matchedCredential));
      }
    }
  }

  private void handleCallExpression(CallExpression callExpression, SubscriptionContext ctx) {
    if (callExpression.arguments().isEmpty()) {
      return;
    }
    Symbol calleeSymbol = callExpression.calleeSymbol();
    if (calleeSymbol != null && sensitiveArgumentByFQN().containsKey(calleeSymbol.fullyQualifiedName())) {
      checkSensitiveArgument(callExpression, sensitiveArgumentByFQN().get(calleeSymbol.fullyQualifiedName()), ctx);
    }
  }

  private static void checkSensitiveArgument(CallExpression callExpression, int argNb, SubscriptionContext ctx) {
    for (int i = 0; i < callExpression.arguments().size(); i++) {
      Argument argument = callExpression.arguments().get(i);
      if (argument.is(Kind.REGULAR_ARGUMENT)) {
        RegularArgument regularArgument = (RegularArgument) argument;
        if (regularArgument.keywordArgument() != null) {
          return;
        } else if (i == argNb && regularArgument.expression().is(Kind.STRING_LITERAL) && !((StringLiteral) regularArgument.expression()).trimmedQuotesValue().isEmpty()) {
          ctx.addIssue(regularArgument, "Review this potentially hard-coded credential.");
        }
      }
    }
  }

  private void handleStringLiteral(StringLiteral stringLiteral, SubscriptionContext ctx) {
    if (isDocString(stringLiteral)) {
      return;
    }
    if (stringLiteral.stringElements().stream().anyMatch(StringElement::isInterpolated)) {
      return;
    }

    String literalValue = stringLiteral.trimmedQuotesValue();

    if (isFlaskConfigAssignment(stringLiteral)) {
      if (FLASK_CONFIG_CREDENTIAL_KEY.equals(literalValue)) {
        ctx.addIssue(stringLiteral, String.format(MESSAGE, FLASK_CONFIG_CREDENTIAL_KEY));
      }
      return;
    }

    String matchedCredential = matchedCredential(literalValue, literalPatterns());
    if (matchedCredential != null) {
      ctx.addIssue(stringLiteral, String.format(MESSAGE, matchedCredential));
    }
    if (isURLWithCredentials(stringLiteral)) {
      ctx.addIssue(stringLiteral, "Review this hard-coded URL, which may contain a credential.");
    }
  }

  /**
   * Flask config assignment should not raise an issue except an assignment to `SECRET_KEY`
   * See SONARPY-1061 for more context
   */
  private static boolean isFlaskConfigAssignment(Expression expression) {
    if (expression.parent().is(Kind.ASSIGNMENT_STMT)) {
      AssignmentStatement assignment = (AssignmentStatement) expression.parent();
      return assignment.lhsExpressions().stream()
        .map(ExpressionList::expressions)
        .flatMap(Collection::stream)
        .anyMatch(HardCodedCredentialsCheck::isFlaskConfigSubscription);
    }
    return Optional.ofNullable(TreeUtils.firstAncestorOfKind(expression, Kind.SUBSCRIPTION))
      .filter(HardCodedCredentialsCheck::isFlaskConfigSubscription)
      .isPresent();
  }

  private static boolean isFlaskConfigSubscription(Tree tree) {
    if (tree.is(Kind.SUBSCRIPTION)) {
      return getSubscriptionFqn((SubscriptionExpression) tree)
        .filter(FLASK_CONFIG_ASSIGNMENT_FQN::equals)
        .isPresent();
    }
    return false;
  }

  private static Optional<String> getSubscriptionFqn(SubscriptionExpression subscription) {
    return Optional.of(subscription.object())
      .filter(QualifiedExpression.class::isInstance)
      .map(qualifier -> ((QualifiedExpression) qualifier).name().symbol())
      .map(Symbol::fullyQualifiedName);
  }

  private static boolean isDocString(StringLiteral stringLiteral) {
    Tree parent = TreeUtils.firstAncestorOfKind(stringLiteral, Kind.FILE_INPUT, Kind.CLASSDEF, Kind.FUNCDEF);
    return Optional.ofNullable(parent)
      .map(p -> ((p.is(Kind.FILE_INPUT) && stringLiteral.equals(((FileInput) p).docstring()))
        || (p.is(Kind.CLASSDEF) && stringLiteral.equals(((ClassDef) p).docstring()))
        || (p.is(Kind.FUNCDEF) && stringLiteral.equals(((FunctionDef) p).docstring()))))
      .orElse(false);
  }

  private static boolean isURLWithCredentials(StringLiteral stringLiteral) {
    if (!stringLiteral.trimmedQuotesValue().contains("://")) {
      return false;
    }
    try {
      URL url = new URL(stringLiteral.trimmedQuotesValue());
      String userInfo = url.getUserInfo();
      if (userInfo != null && userInfo.matches("\\S+:\\S+")) {
        return true;
      }
    } catch (MalformedURLException e) {
      return false;
    }
    return false;
  }

  private void handleAssignmentStatement(AssignmentStatement assignmentStatement, SubscriptionContext ctx) {
    ExpressionList lhs = assignmentStatement.lhsExpressions().get(0);
    Expression expression = lhs.expressions().get(0);

    if (expression instanceof HasSymbol hasSymbol) {
      Symbol symbol = hasSymbol.symbol();
      String matchedCredential = credentialSymbolName(symbol);
      if (matchedCredential != null) {
        checkAssignedValue(assignmentStatement, matchedCredential, ctx);
      }
    }

    if (expression.is(Kind.SUBSCRIPTION)) {
      SubscriptionExpression subscriptionExpression = (SubscriptionExpression) expression;
      for (Expression expr : subscriptionExpression.subscripts().expressions()) {
        if (expr.is(Kind.STRING_LITERAL)) {
          String matchedCredential = matchedCredential(((StringLiteral) expr).trimmedQuotesValue(), variablePatterns());
          if (matchedCredential != null) {
            checkAssignedValue(assignmentStatement, matchedCredential, ctx);
          }
        }
      }
    }
  }

  private void checkAssignedValue(AssignmentStatement assignmentStatement, String matchedCredential, SubscriptionContext ctx) {
    Expression assignedValue = assignmentStatement.assignedValue();
    if (isSuspiciousStringLiteral(assignedValue) && !isFlaskConfigAssignment(assignedValue) ) {
      ctx.addIssue(assignmentStatement, String.format(MESSAGE, matchedCredential));
    }
  }

  private String credentialSymbolName(@CheckForNull Symbol symbol) {
    if (symbol != null) {
      return matchedCredential(symbol.name(), variablePatterns());
    }
    return null;
  }

  private boolean isSuspiciousStringLiteral(Tree tree) {
    return tree.is(Kind.STRING_LITERAL) && !((StringLiteral) tree).trimmedQuotesValue().isEmpty()
      && !isCredential(((StringLiteral) tree).trimmedQuotesValue(), variablePatterns());
  }

  private static boolean isCredential(String target, Stream<Pattern> patterns) {
    return patterns.anyMatch(pattern -> pattern.matcher(target).find());
  }

  private static String matchedCredential(String target, Stream<Pattern> patterns) {
    Optional<Pattern> matched = patterns.filter(pattern -> pattern.matcher(target).find()).findFirst();
    if (matched.isPresent()) {
      String matchedPattern = matched.get().pattern();
      int suffixStart = matchedPattern.indexOf('=');
      if (suffixStart > 0) {
        return matchedPattern.substring(0, suffixStart);
      } else {
        return matchedPattern;
      }
    }
    return null;
  }

  private List<Pattern> toPatterns(String suffix) {
    return Stream.of(credentialWords.split(","))
      .map(String::trim)
      .map(word -> Pattern.compile(word + suffix, Pattern.CASE_INSENSITIVE))
      .toList();
  }
}
