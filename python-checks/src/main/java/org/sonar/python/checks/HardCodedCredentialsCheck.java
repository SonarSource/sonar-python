/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2020 SonarSource SA
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
package org.sonar.python.checks;

import java.net.MalformedURLException;
import java.net.URL;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
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
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.DictionaryLiteral;
import org.sonar.plugins.python.api.tree.DictionaryLiteralElement;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionList;
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

@Rule(key = "S2068")
public class HardCodedCredentialsCheck extends PythonSubscriptionCheck {

  private static final String DEFAULT_CREDENTIAL_WORDS = "password,passwd,pwd, passphrase";

  @RuleProperty(
    key = "credentialWords",
    description = "Comma separated list of words identifying potential credentials",
    defaultValue = DEFAULT_CREDENTIAL_WORDS)
  public String credentialWords = DEFAULT_CREDENTIAL_WORDS;

  public static final String MESSAGE = "Remove this hard-coded password.";

  private List<Pattern> variablePatterns = null;
  private List<Pattern> literalPatterns = null;
  private Map<String, Integer> sensitiveArgumentByFQN;

  private Map<String, Integer> sensitiveArgumentByFQN() {
    if (sensitiveArgumentByFQN == null) {
      sensitiveArgumentByFQN = new HashMap<>();
      sensitiveArgumentByFQN.put("mysql.connector.connect", 2);
      sensitiveArgumentByFQN.put("mysql.connector.connection.MySQLConnection", 2);
      sensitiveArgumentByFQN.put("pymysql.connect", 2);
      sensitiveArgumentByFQN.put("pymysql.connections.Connection", 2);
      sensitiveArgumentByFQN.put("psycopg2.connect", 2);
      sensitiveArgumentByFQN.put("pgdb.connect", 2);
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
      literalPatterns = toPatterns("=[^:%'?\\s]+");
    }
    return literalPatterns.stream();
  }

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Kind.ASSIGNMENT_STMT, ctx -> handleAssignmentStatement((AssignmentStatement) ctx.syntaxNode(), ctx));
    context.registerSyntaxNodeConsumer(Kind.COMPARISON, ctx -> handleBinaryExpression((BinaryExpression) ctx.syntaxNode(), ctx));
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
        if (keyValuePair.key().is(Kind.STRING_LITERAL) && isCredential(((StringLiteral) keyValuePair.key()).trimmedQuotesValue(), variablePatterns())
          && keyValuePair.value().is(Kind.STRING_LITERAL)) {
          ctx.addIssue(dictionaryLiteralElement, MESSAGE);
        }
      }
    }
  }

  private void handleParameterList(ParameterList parameterList, SubscriptionContext ctx) {
    for (Parameter parameter : parameterList.nonTuple()) {
      Name parameterName = parameter.name();
      Expression defaultValue = parameter.defaultValue();
      if (parameterName != null && isCredential(parameterName.name(), variablePatterns()) && defaultValue != null
        && isNonEmptyStringLiteral(defaultValue)) {
        ctx.addIssue(parameter, MESSAGE);
      }
    }
  }

  private void handleRegularArgument(RegularArgument regularArgument, SubscriptionContext ctx) {
    Name keywordArgument = regularArgument.keywordArgument();
    if (keywordArgument != null && isCredential(keywordArgument.name(), variablePatterns())
      && regularArgument.expression().is(Kind.STRING_LITERAL) && !isEmptyStringLiteral((StringLiteral) regularArgument.expression())) {
      ctx.addIssue(regularArgument, MESSAGE);
    }
  }

  private void handleCallExpression(CallExpression callExpression, SubscriptionContext ctx) {
    if (callExpression.arguments().isEmpty()) {
      return;
    }
    // Raising issues on pwd.__eq__("literal") calls
    if (callExpression.callee().is(Kind.QUALIFIED_EXPR)) {
      QualifiedExpression qualifiedExpression = (QualifiedExpression) callExpression.callee();
      if (qualifiedExpression.name().name().equals("__eq__")) {
        if (isQualifiedExpressionCredential(qualifiedExpression)) {
          if (isFirstArgumentAStringLiteral(callExpression)) {
            ctx.addIssue(callExpression, MESSAGE);
          }
        } else if (qualifiedExpression.qualifier().is(Kind.STRING_LITERAL) && isFirstArgumentCredentials(callExpression)) {
          ctx.addIssue(callExpression, MESSAGE);
        }
      }
    }
    Symbol calleeSymbol = callExpression.calleeSymbol();
    if (calleeSymbol != null && sensitiveArgumentByFQN().containsKey(calleeSymbol.fullyQualifiedName())) {
      checkSensitiveArgument(callExpression, sensitiveArgumentByFQN().get(calleeSymbol.fullyQualifiedName()), ctx);
    }
  }

  private boolean isFirstArgumentCredentials(CallExpression callExpression) {
    Argument argument = callExpression.arguments().get(0);
    if (argument.is(Kind.REGULAR_ARGUMENT)) {
      RegularArgument regularArgument = (RegularArgument) argument;
      return regularArgument.expression().is(Kind.NAME) && isCredential(((Name) regularArgument.expression()).name(), variablePatterns());
    }
    return false;
  }

  private static boolean isFirstArgumentAStringLiteral(CallExpression callExpression) {
    return callExpression.arguments().get(0).is(Kind.REGULAR_ARGUMENT)
      && ((RegularArgument) callExpression.arguments().get(0)).expression().is(Kind.STRING_LITERAL);
  }

  private boolean isQualifiedExpressionCredential(QualifiedExpression qualifiedExpression) {
    return qualifiedExpression.qualifier().is(Kind.NAME) && isCredential(((Name) qualifiedExpression.qualifier()).name(), variablePatterns());
  }

  private static void checkSensitiveArgument(CallExpression callExpression, int argNb, SubscriptionContext ctx) {
    for (int i = 0; i < callExpression.arguments().size(); i++) {
      Argument argument = callExpression.arguments().get(i);
      if (argument.is(Kind.REGULAR_ARGUMENT)) {
        RegularArgument regularArgument = (RegularArgument) argument;
        if (regularArgument.keywordArgument() != null) {
          return;
        } else if (i == argNb && regularArgument.expression().is(Kind.STRING_LITERAL)) {
          ctx.addIssue(callExpression, MESSAGE);
        }
      }
    }
  }

  private void handleStringLiteral(StringLiteral stringLiteral, SubscriptionContext ctx) {
    if (stringLiteral.stringElements().stream().anyMatch(StringElement::isInterpolated)) {
      return;
    }
    if (isCredential(stringLiteral.trimmedQuotesValue(), literalPatterns())) {
      ctx.addIssue(stringLiteral, MESSAGE);
    }
    if (isURLWithCredentials(stringLiteral)) {
      ctx.addIssue(stringLiteral, MESSAGE);
    }
  }

  private static boolean isURLWithCredentials(StringLiteral stringLiteral) {
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

  private void handleBinaryExpression(BinaryExpression binaryExpression, SubscriptionContext ctx) {
    boolean shouldReport = false;
    if (binaryExpression.leftOperand() instanceof HasSymbol && binaryExpression.rightOperand().is(Tree.Kind.STRING_LITERAL)) {
      shouldReport = isSymbolCredential(((HasSymbol) binaryExpression.leftOperand()).symbol());
    }
    if (binaryExpression.rightOperand() instanceof HasSymbol && binaryExpression.leftOperand().is(Tree.Kind.STRING_LITERAL)) {
      shouldReport = isSymbolCredential(((HasSymbol) binaryExpression.rightOperand()).symbol());
    }
    if (shouldReport) {
      ctx.addIssue(binaryExpression, MESSAGE);
    }
  }

  private void handleAssignmentStatement(AssignmentStatement assignmentStatement, SubscriptionContext ctx) {
    ExpressionList lhs = assignmentStatement.lhsExpressions().get(0);
    Expression expression = lhs.expressions().get(0);

    if (expression instanceof HasSymbol) {
      Symbol symbol = ((HasSymbol) expression).symbol();
      if (isSymbolCredential(symbol)) {
        checkAssignedValue(assignmentStatement, ctx);
      }
    }

    if (expression.is(Kind.SUBSCRIPTION)) {
      SubscriptionExpression subscriptionExpression = (SubscriptionExpression) expression;
      if (subscriptionExpression.subscripts().expressions().stream().anyMatch(e -> e.is(Kind.STRING_LITERAL) &&
        isCredential(((StringLiteral) e).trimmedQuotesValue(), variablePatterns()))) {
        checkAssignedValue(assignmentStatement, ctx);
      }
    }
  }

  private void checkAssignedValue(AssignmentStatement assignmentStatement, SubscriptionContext ctx) {
    if (assignmentStatement.assignedValue().is(Kind.STRING_LITERAL)) {
      StringLiteral literal = (StringLiteral) assignmentStatement.assignedValue();
      if (!isEmptyStringLiteral(literal) && !isCredential(literal.trimmedQuotesValue(), variablePatterns())) {
        ctx.addIssue(assignmentStatement, MESSAGE);
      }
    }
  }

  private boolean isSymbolCredential(@CheckForNull Symbol symbol) {
    return symbol != null && isCredential(symbol.name(), variablePatterns());
  }

  private static boolean isNonEmptyStringLiteral(Tree tree) {
    return tree.is(Kind.STRING_LITERAL) && !isEmptyStringLiteral((StringLiteral) tree);
  }

  private static boolean isEmptyStringLiteral(StringLiteral stringLiteral) {
    return stringLiteral.trimmedQuotesValue().isEmpty();
  }

  private static boolean isCredential(String target, Stream<Pattern> patterns) {
    return patterns.anyMatch(pattern -> pattern.matcher(target).find());
  }

  private List<Pattern> toPatterns(String suffix) {
    return Stream.of(credentialWords.split(","))
      .map(String::trim)
      .map(word -> Pattern.compile(word + suffix, Pattern.CASE_INSENSITIVE))
      .collect(Collectors.toList());
  }
}
