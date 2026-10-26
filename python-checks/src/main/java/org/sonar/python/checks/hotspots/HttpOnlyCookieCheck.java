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

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.DictionaryLiteral;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.KeyValuePair;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.SubscriptionExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.utils.Expressions;

@Rule(key = "S3330")
public class HttpOnlyCookieCheck extends AbstractCookieFlagCheck {

  public static final String HTTPONLY_ARGUMENT_NAME = "httponly";
  public static final String HEADERS_ARGUMENT_NAME = "headers";

  private final MethodArgumentsToCheckRegistry methodArgumentsToCheckRegistry = createMethodArgumentsToCheckRegistry();

  private MethodArgumentsToCheckRegistry createMethodArgumentsToCheckRegistry() {
    var methods = new ArrayList<>(List.of(
      new MethodArgumentsToCheck(CookieApi.DJANGO_RESPONSE, CookieApi.SET_COOKIE_METHOD_NAME, HTTPONLY_ARGUMENT_NAME, 7),
      new MethodArgumentsToCheck(CookieApi.DJANGO_RESPONSE, CookieApi.SET_SIGNED_COOKIE_METHOD_NAME, HTTPONLY_ARGUMENT_NAME, 8),
      new MethodArgumentsToCheck(CookieApi.FLASK_RESPONSE, CookieApi.SET_COOKIE_METHOD_NAME, HTTPONLY_ARGUMENT_NAME, 7),
      new MethodArgumentsToCheck(CookieApi.WERKZEUG_BASE_RESPONSE, CookieApi.SET_COOKIE_METHOD_NAME, HTTPONLY_ARGUMENT_NAME, 7),
      new MethodArgumentsToCheck(CookieApi.WERKZEUG_RESPONSE, CookieApi.SET_COOKIE_METHOD_NAME, HTTPONLY_ARGUMENT_NAME, 7)));
    for (String responseType : CookieApi.ASGI_RESPONSE_TYPES) {
      methods.add(new MethodArgumentsToCheck(responseType, CookieApi.SET_COOKIE_METHOD_NAME, HTTPONLY_ARGUMENT_NAME, -1));
      methods.add(new MethodArgumentsToCheck(responseType, HEADERS_ARGUMENT_NAME, -1, this::isInvalidHeaderArgument));
    }
    return new MethodArgumentsToCheckRegistry(methods.toArray(MethodArgumentsToCheck[]::new));
  }

  @Override
  String flagName() {
    return HTTPONLY_ARGUMENT_NAME;
  }

  @Override
  String message() {
    return "Make sure creating this cookie without the \"HttpOnly\" flag is safe.";
  }

  @Override
  MethodArgumentsToCheckRegistry methodArgumentsToCheckRegistry() {
    return methodArgumentsToCheckRegistry;
  }


  @Override
  public void initialize(Context context) {
    super.initialize(context);
    context.registerSyntaxNodeConsumer(
      Tree.Kind.ASSIGNMENT_STMT,
      this::subscriptionSessionCookieHttponlyCheck
    );
    context.registerSyntaxNodeConsumer(
      Tree.Kind.DICTIONARY_LITERAL,
      this::dictionarySessionCookieHttponlyCheck
    );
    context.registerSyntaxNodeConsumer(
      Tree.Kind.CALL_EXPR,
      this::dictConstructorSessionCookieHttponlyCheck
    );
  }

  @Override
  protected String headerValueRegex() {
    return ".*;\\s?HttpOnly.*";
  }

  private void subscriptionSessionCookieHttponlyCheck(SubscriptionContext ctx) {
    AssignmentStatement assignmentStatement = (AssignmentStatement) ctx.syntaxNode();
    boolean isSubscriptionToSessionCookieHttponly = assignmentStatement
      .lhsExpressions()
      .stream()
      .flatMap(exprList -> exprList.expressions().stream())
      .filter(expr -> expr.is(Tree.Kind.SUBSCRIPTION))
      .flatMap(subscription -> ((SubscriptionExpression) subscription).subscripts().expressions().stream())
      .anyMatch(HttpOnlyCookieCheck::isSessionCookieHttponlyStringLiteral);
    if (isSubscriptionToSessionCookieHttponly && Expressions.isFalsy(assignmentStatement.assignedValue())) {
      ctx.addIssue(assignmentStatement.assignedValue(), message());
    }
  }

  private void dictionarySessionCookieHttponlyCheck(SubscriptionContext ctx) {
    DictionaryLiteral dict = (DictionaryLiteral) ctx.syntaxNode();
    Optional<Expression> falsySetting = searchForFalsySessionCookieHttponlyInDictionary(dict);
    falsySetting.ifPresent(expression -> ctx.addIssue(expression, message()));
  }

  private static Optional<Expression> searchForFalsySessionCookieHttponlyInDictionary(DictionaryLiteral dict) {
    return dict.elements().stream()
      .filter(e -> e.is(Tree.Kind.KEY_VALUE_PAIR))
      .map(KeyValuePair.class::cast)
      .filter(kvp -> Optional.ofNullable(kvp.key())
        .filter(HttpOnlyCookieCheck::isSessionCookieHttponlyStringLiteral)
        .isPresent())
      .findFirst()
      .filter(kvp -> Expressions.isFalsy(kvp.value()))
      .map(KeyValuePair::value);
  }

  private static final String SESSION_COOKIE_HTTPONLY = "SESSION_COOKIE_HTTPONLY";

  private void dictConstructorSessionCookieHttponlyCheck(SubscriptionContext ctx) {
    CallExpression callExpression = (CallExpression) ctx.syntaxNode();
    Optional<Expression> falsySetting = searchForFalsySessionCookieHttponlyInDictCons(callExpression);
    falsySetting.ifPresent(expression -> ctx.addIssue(expression, message()));
  }

  private static Optional<Expression> searchForFalsySessionCookieHttponlyInDictCons(CallExpression callExpression) {
    Symbol callee = callExpression.calleeSymbol();
    if (callee != null && "dict".equals(callee.fullyQualifiedName())) {
      for (Argument arg: callExpression.arguments()) {
        if (arg.is(Tree.Kind.REGULAR_ARGUMENT)) {
          RegularArgument regArg = (RegularArgument) arg;
          Name key = regArg.keywordArgument();
          if (
            key != null &&
              SESSION_COOKIE_HTTPONLY.equals(key.name()) &&
              Expressions.isFalsy(regArg.expression())
          ) {
            return Optional.of(regArg.expression());
          }
        }
      }
    }
    return Optional.empty();
  }

  private static boolean isSessionCookieHttponlyStringLiteral(Expression expr) {
    return
      expr.is(Tree.Kind.STRING_LITERAL) &&
        SESSION_COOKIE_HTTPONLY.equals(((StringLiteral) expr).trimmedQuotesValue());
  }
}
