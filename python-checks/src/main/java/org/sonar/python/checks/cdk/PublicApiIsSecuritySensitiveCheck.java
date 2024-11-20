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
package org.sonar.python.checks.cdk;

import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.function.Predicate;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonVisitorContext;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.FunctionDefImpl;
import org.sonar.python.tree.TreeUtils;

import static org.sonar.python.checks.cdk.CdkPredicate.isFqn;
import static org.sonar.python.checks.cdk.CdkPredicate.isString;
import static org.sonar.python.checks.cdk.CdkUtils.getArgument;
import static org.sonar.python.checks.cdk.CdkUtils.getDictionary;
import static org.sonar.python.checks.cdk.CdkUtils.getDictionaryPair;

@Rule(key = "S6333")
public class PublicApiIsSecuritySensitiveCheck extends AbstractCdkResourceCheck {
  private static final String MESSAGE = "Make sure that creating public APIs is safe here.";
  private static final String OMITTING_MESSAGE = "Omitting \"authorization_type\" disables authentication. Make sure it is safe here.";
  private static final String AUTHORIZATION_TYPE = "authorization_type";
  private static final String AUTHORIZATION_TYPE_NONE = "aws_cdk.aws_apigateway.AuthorizationType.NONE";

  private final Set<String> safeMethods = new HashSet<>();

  @Override
  public void scanFile(PythonVisitorContext visitorContext) {
    super.scanFile(visitorContext);
    safeMethods.clear();
  }

  @Override
  protected void registerFqnConsumer() {
    checkFqns(List.of("aws_cdk.aws_apigateway.CfnMethod", "aws_cdk.aws_apigatewayv2.CfnRoute"), (subscriptionContext, callExpression) ->
      getArgument(subscriptionContext, callExpression, AUTHORIZATION_TYPE).ifPresentOrElse(
        argument -> argument.addIssueIf(isString("NONE"), MESSAGE),
        () -> subscriptionContext.addIssue(callExpression.callee(), OMITTING_MESSAGE)
      )
    );

    checkFqns(List.of("aws_cdk.aws_apigateway.RestApi", "aws_cdk.aws_apigateway.Resource.add_resource"), (subscriptionContext, callExpression) ->
      getArgument(subscriptionContext, callExpression, "default_method_options").ifPresent(
        argument -> {
          if (isArgumentSafe(subscriptionContext, argument)) {
            // if invocation of RestApi() or Resource.add_resource() is with the safe default, and it's in the method,
            // then store the method's full qualified name in safeMethods
            enclosingMethodFqn(callExpression).ifPresent(safeMethods::add);
          }
        }
      )
    );

    checkFqn("aws_cdk.aws_apigateway.Resource.add_method", (subscriptionContext, callExpression) ->
      getArgument(subscriptionContext, callExpression, AUTHORIZATION_TYPE).ifPresentOrElse(
        argument -> argument.addIssueIf(isFqn(AUTHORIZATION_TYPE_NONE), MESSAGE),
        () -> enclosingMethodFqn(callExpression).filter(fqn -> !safeMethods.contains(fqn)).ifPresent(fnq ->
          // if the invocation of Resource.add_method() is without explicit authorization_type argument,
          // and it's not in the scope of safeMethod the call, then it's considered unsafe
          // this is the best possible approximation for now
          subscriptionContext.addIssue(callExpression.callee(), OMITTING_MESSAGE)
        )
      )
    );
  }

  private static Optional<String> enclosingMethodFqn(Tree tree) {
    FunctionDef functionDef = (FunctionDef) TreeUtils.firstAncestorOfKind(tree, Tree.Kind.FUNCDEF);
    return Optional.ofNullable(functionDef)
      .map(FunctionDefImpl.class::cast)
      .map(FunctionDefImpl::functionSymbol)
      .map(Symbol::fullyQualifiedName);
  }

  private static boolean isArgumentSafe(SubscriptionContext subscriptionContext, CdkUtils.ExpressionFlow argument) {
    return !(isUnsafeSafeDictionaryAuthorisationKey(subscriptionContext, argument.getLast())
      || isUnsafeAuthorisationArgument(subscriptionContext, argument));
  }

  private static boolean isUnsafeSafeDictionaryAuthorisationKey(SubscriptionContext ctx, Expression expression) {
    return getDictionary(expression)
      .flatMap(dictionary -> getDictionaryPair(ctx, dictionary, AUTHORIZATION_TYPE))
      .filter(element -> isFqn(AUTHORIZATION_TYPE_NONE).test(element.value.getLast()))
      .isPresent();
  }

  private static boolean isUnsafeAuthorisationArgument(SubscriptionContext subscriptionContext, CdkUtils.ExpressionFlow expression) {
    return expression.getExpression(isCallExpression().and(isFqn("aws_cdk.aws_apigateway.MethodOptions")))
      .flatMap(expr -> getArgument(subscriptionContext, (CallExpression) expr, AUTHORIZATION_TYPE))
      .filter(expr -> expr.hasExpression(isFqn(AUTHORIZATION_TYPE_NONE)))
      .isPresent();
  }

  public static Predicate<Expression> isCallExpression() {
    return expression -> expression.is(Tree.Kind.CALL_EXPR);
  }
}
