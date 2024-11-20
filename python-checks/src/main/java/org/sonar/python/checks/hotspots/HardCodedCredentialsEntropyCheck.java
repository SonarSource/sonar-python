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
package org.sonar.python.checks.hotspots;

import java.util.ArrayList;
import java.util.Collection;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import org.sonar.check.Rule;
import org.sonar.check.RuleProperty;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.AnnotatedAssignment;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.DictionaryLiteral;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.KeyValuePair;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.ParameterList;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.utils.Expressions;
import org.sonarsource.analyzer.commons.ShannonEntropy;

@Rule(key = "S6418")
public class HardCodedCredentialsEntropyCheck extends PythonSubscriptionCheck {

  private static final String DEFAULT_SECRET_KEYWORDS = "api[_.-]?key,auth,credential,secret,token";

  private static final String DEFAULT_RANDOMNESS_SENSIBILITY = "3.0";

  private static final Pattern POSTVALIDATION_PATTERN = Pattern.compile("[a-zA-Z0-9_.+/~$-]([a-zA-Z0-9_.+/=~$-]|\\\\\\\\(?![ntr\"])){14,1022}[a-zA-Z0-9_.+/=~$-]");

  private static final String MESSAGE = "\"%s\" detected here, make sure this is not a hard-coded secret.";

  private Collection<Pattern> patterns = null;

  @RuleProperty(
    key = "credentialWords",
    description = "Comma separated list of words identifying potential credentials",
    defaultValue = DEFAULT_SECRET_KEYWORDS)
  public String secretKeyWords = DEFAULT_SECRET_KEYWORDS;

  @RuleProperty(
    key = "randomnessSensibility",
    description = "Allows to tune the Randomness Sensibility (from 0 to 10)",
    defaultValue = DEFAULT_RANDOMNESS_SENSIBILITY)
  public double randomnessSensibility = Double.parseDouble(DEFAULT_RANDOMNESS_SENSIBILITY);

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.ASSIGNMENT_STMT, this::checkAssignment);
    context.registerSyntaxNodeConsumer(Tree.Kind.ANNOTATED_ASSIGNMENT, this::checkAnnotatedAssignment);
    context.registerSyntaxNodeConsumer(Tree.Kind.PARAMETER_LIST, this::checkParameterList);
    context.registerSyntaxNodeConsumer(Tree.Kind.REGULAR_ARGUMENT, this::checkRegularArgument);
    context.registerSyntaxNodeConsumer(Tree.Kind.DICTIONARY_LITERAL, this::checkDictionaryLiteral);
  }

  private void patternMatch(Name name, Tree location, String value, SubscriptionContext subscriptionContext) {
    patternMatch(name.name(), location, value, subscriptionContext);
  }

  private void patternMatch(String name, Tree location, String value, SubscriptionContext subscriptionContext) {
    if (!valuePassesPostValidation(value) || !entropyShouldRaise(value)) {
      return;
    }
    patterns().stream()
      .filter(pattern -> pattern.matcher(name).matches())
      .findFirst()
      .ifPresent(pattern -> subscriptionContext.addIssue(location, String.format(MESSAGE, name)));
  }

  private void checkParameterList(SubscriptionContext subscriptionContext) {
    var parameterList = (ParameterList) subscriptionContext.syntaxNode();
    parameterList.nonTuple().stream()
      .filter(parameter -> parameter.name() != null)
      .filter(parameter -> parameter.defaultValue() != null)
      .filter(parameter -> parameter.defaultValue().is(Tree.Kind.STRING_LITERAL))
      .forEach(parameter -> {
        var value = ((StringLiteral) parameter.defaultValue()).trimmedQuotesValue();
        patternMatch(parameter.name(), parameter.defaultValue(), value, subscriptionContext);
      });
  }

  private void checkDictionaryLiteral(SubscriptionContext subscriptionContext) {
    var dictionaryLiteral = (DictionaryLiteral) subscriptionContext.syntaxNode();
    dictionaryLiteral.elements().stream().filter(e -> e.is(Tree.Kind.KEY_VALUE_PAIR))
      .map(KeyValuePair.class::cast)
      .filter(keyValuePair -> keyValuePair.value().is(Tree.Kind.STRING_LITERAL))
      .filter(keyValuePair -> keyValuePair.key().is(Tree.Kind.STRING_LITERAL))
      .forEach(keyValuePair -> {
        var value = ((StringLiteral) keyValuePair.value()).trimmedQuotesValue();
        var key = ((StringLiteral) keyValuePair.key()).trimmedQuotesValue();
        patternMatch(key, keyValuePair.value(), value, subscriptionContext);
      });
  }

  private void checkRegularArgument(SubscriptionContext subscriptionContext) {
    var regularArgument = (RegularArgument) subscriptionContext.syntaxNode();
    var keywordArgument = regularArgument.keywordArgument();

    if (keywordArgument != null && regularArgument.expression() instanceof StringLiteral expression) {
      var value = expression.trimmedQuotesValue();
      patternMatch(keywordArgument, regularArgument, value, subscriptionContext);
    }
  }

  private void checkAnnotatedAssignment(SubscriptionContext subscriptionContext) {
    var annotatedAssignment = (AnnotatedAssignment) subscriptionContext.syntaxNode();
    var assignedValue = annotatedAssignment.assignedValue();
    var assignedVariable = annotatedAssignment.variable();
    if (assignedValue instanceof StringLiteral stringLiteral && assignedVariable instanceof Name name) {
      patternMatch(name, assignedValue, stringLiteral.trimmedQuotesValue(), subscriptionContext);
    }
  }

  private void checkAssignment(SubscriptionContext subscriptionContext) {
    var expressions = new ArrayList<Expression>();
    var assignment = (AssignmentStatement) subscriptionContext.syntaxNode();
    var assignedValue = assignment.assignedValue();

    if (assignedValue.is(Tree.Kind.TUPLE)) {
      expressions.addAll(Expressions.getExpressionsFromRhs(assignedValue));
    } else {
      expressions.add(assignedValue);
    }

    expressions.stream()
      .filter(StringLiteral.class::isInstance)
      .map(StringLiteral.class::cast)
      .forEach(expression -> Expressions.getAssignedName(expression)
        .ifPresent(name -> {
          var value = expression.trimmedQuotesValue();
          patternMatch(name, expression, value, subscriptionContext);
        }));
  }

  private static boolean valuePassesPostValidation(String value) {
    return POSTVALIDATION_PATTERN.matcher(value).matches();
  }

  private boolean entropyShouldRaise(String value) {
    return ShannonEntropy.calculate(value) > randomnessSensibility;
  }

  private Collection<Pattern> patterns() {
    if (patterns == null) {
      patterns = Stream.of(secretKeyWords.split(","))
        .map(word -> Pattern.compile("(" + word + ")", Pattern.CASE_INSENSITIVE))
        .toList();
    }
    return patterns;
  }

}
