/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */
package org.sonar.python.checks.hotspots;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.check.RuleProperty;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
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

@Rule(key = "S6418")
public class HardCodedCredentialsEntropyCheck extends PythonSubscriptionCheck {

  private static final String DEFAULT_SECRET_KEYWORDS = "api[_.-]?key,auth,credential,secret,token";

  @RuleProperty(
    key = "credentialWords",
    description = "Comma separated list of words identifying potential credentials",
    defaultValue = DEFAULT_SECRET_KEYWORDS)
  public String secretKeyWords = DEFAULT_SECRET_KEYWORDS;

  private Collection<Pattern> patterns = null;

  private static final String DEFAULT_RANDOMNESS_SENSIBILITY = "5.0";
  @RuleProperty(
    key = "randomnessSensibility",
    description = "Allows to tune the Randomness Sensibility (from 0 to 10)",
    defaultValue = DEFAULT_RANDOMNESS_SENSIBILITY)
  public double randomnessSensibility = Double.parseDouble(DEFAULT_RANDOMNESS_SENSIBILITY);

  private static final Pattern POSTVALIDATION_PATTERN = Pattern.compile("[a-zA-Z0-9_.+/~$-]([a-zA-Z0-9_.+/=~$-]|\\\\\\\\(?![ntr\"])){14,1022}[a-zA-Z0-9_.+/=~$-]");

  private static final String MESSAGE = "\"%s\" detected here, make sure this is not a hard-coded secret.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.ASSIGNMENT_STMT, this::checkAssignment);
    context.registerSyntaxNodeConsumer(Tree.Kind.PARAMETER_LIST, this::checkParameterList);
    context.registerSyntaxNodeConsumer(Tree.Kind.REGULAR_ARGUMENT, this::checkRegularArgument);
    context.registerSyntaxNodeConsumer(Tree.Kind.DICTIONARY_LITERAL, this::checkDictionaryLiteral);
  }

  private void patternMatch(Name name, Tree location, SubscriptionContext subscriptionContext) {
    patternMatch(name.name(), location, subscriptionContext);
  }

  private void patternMatch(String name, Tree location, SubscriptionContext subscriptionContext) {
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
      .filter(parameter -> {
        var value = ((StringLiteral) parameter.defaultValue()).trimmedQuotesValue();
        return valuePassesPostValidation(value) && entropyShouldRaise(value);
      })
      .forEach(parameter -> patternMatch(parameter.name(), parameter.defaultValue(), subscriptionContext));
  }

  private void checkDictionaryLiteral(SubscriptionContext subscriptionContext) {
    var dictionaryLiteral = (DictionaryLiteral) subscriptionContext.syntaxNode();
    dictionaryLiteral.elements().stream().filter(e -> e.is(Tree.Kind.KEY_VALUE_PAIR))
      .map(KeyValuePair.class::cast)
      .filter(keyValuePair -> keyValuePair.value().is(Tree.Kind.STRING_LITERAL))
      .forEach(keyValuePair -> {
        var value = ((StringLiteral) keyValuePair.value()).trimmedQuotesValue();
        var key = keyValuePair.key();
        if (!key.is(Tree.Kind.STRING_LITERAL)) {
          return;
        }
        if (valuePassesPostValidation(value) && entropyShouldRaise(value)) {
          patternMatch(((StringLiteral) key).trimmedQuotesValue(), keyValuePair.value(), subscriptionContext);
        }
      });
  }

  private void checkRegularArgument(SubscriptionContext subscriptionContext) {
    var regularArgument = (RegularArgument) subscriptionContext.syntaxNode();
    var keywordArgument = regularArgument.keywordArgument();
    if (keywordArgument == null) {
      return;
    }
    var a = regularArgument.expression();
    if (!a.is(Tree.Kind.STRING_LITERAL)) {
      return;
    }
    var value = ((StringLiteral) a).trimmedQuotesValue();
    if (!valuePassesPostValidation(value) || !entropyShouldRaise(value)) {
      return;
    }
    patternMatch(keywordArgument, regularArgument, subscriptionContext);
  }

  private void checkAssignment(SubscriptionContext subscriptionContext) {
    var assignment = (AssignmentStatement) subscriptionContext.syntaxNode();
    var assignedValue = assignment.assignedValue();
    Map<Expression, Name> expressions = null;
    if (assignedValue.is(Tree.Kind.TUPLE)) {
      expressions = Expressions.getExpressionsFromRhs((assignment.assignedValue()))
        .stream()
        .collect(HashMap::new, (map, e) -> {
          var name = Expressions.getAssignedName(e);
          name.ifPresent(value -> map.put(e, value));
        }, HashMap::putAll);
    } else {
      var name = Expressions.getAssignedName(assignedValue);
      if (name.isPresent()) {
        expressions = Map.of(assignedValue, name.get());
      }
    }

    if (expressions == null) {
      return;
    }

    expressions.entrySet().stream()
      .filter(entry -> entry.getKey().is(Tree.Kind.STRING_LITERAL))
      .forEach(entry -> {
        var expression = entry.getKey();
        var name = entry.getValue();
        var value = ((StringLiteral) expression).trimmedQuotesValue();
        if (valuePassesPostValidation(value) && entropyShouldRaise(value)) {
          patternMatch(name, expression, subscriptionContext);
        }
      });

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

  private static final class ShannonEntropy {

    private static final double LOG_2 = Math.log(2.0d);

    private ShannonEntropy() {
      // utility class
    }

    public static double calculate(@Nullable String str) {
      if (str == null || str.isEmpty()) {
        return 0.0d;
      }
      int length = str.length();
      return str.chars()
        .collect(HashMap<Integer, Integer>::new, (map, ch) -> map.merge(ch, 1, Integer::sum), HashMap::putAll)
        .values().stream()
        .mapToDouble(count -> ((double) count) / length)
        .map(frequency -> -(frequency * Math.log(frequency) / LOG_2))
        .sum();
    }
  }

}
