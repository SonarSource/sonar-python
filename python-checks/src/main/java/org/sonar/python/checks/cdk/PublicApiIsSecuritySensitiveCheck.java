/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2022 SonarSource SA
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
package org.sonar.python.checks.cdk;

import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.function.Predicate;
import org.sonar.check.Rule;
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
  public static final String AUTHORIZATION_TYPE_NONE = "aws_cdk.aws_apigateway.AuthorizationType.NONE";

  private Set<String> safeMethods = new HashSet<>();

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
          if (isNotSetToNone(subscriptionContext, argument)) {
            // if invocation of RestApi() or Resource.add_resource() is with the safe default, and it's in method,
            // then the method's full qualified name is stored in safeMethods
            // in current scanner implementation it's impossible to backtrack all previous method executions
            enclosingMethodFqn(callExpression).ifPresent(fqn -> safeMethods.add(fqn));
          }
        }
      )
    );
    checkFqn("aws_cdk.aws_apigateway.Resource.add_method", (subscriptionContext, callExpression) ->
      enclosingMethodFqn(callExpression).filter(fqn -> safeMethods.contains(fqn)).ifPresentOrElse(fqn -> {}, () ->
        getArgument(subscriptionContext, callExpression, AUTHORIZATION_TYPE).ifPresentOrElse(
          argument -> argument.addIssueIf(isFqn(AUTHORIZATION_TYPE_NONE), MESSAGE),
          // if the invocation of Resource.add_method() is not in the scope of safeMethod
          // the call without explicit authorization_type argument is considered unsafe
          () -> subscriptionContext.addIssue(callExpression.callee(), OMITTING_MESSAGE)
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

  private static boolean isNotSetToNone(SubscriptionContext subscriptionContext, CdkUtils.ExpressionFlow argument) {
    return isSafeDictionaryAuthorisationKey(subscriptionContext, argument.getLast())
      || isSafeAuthorisationArgument(subscriptionContext, argument);
  }


  private static boolean isSafeDictionaryAuthorisationKey(SubscriptionContext ctx, Expression expression) {
    return getDictionary(expression)
      .flatMap(dictionary -> getDictionaryPair(ctx, dictionary, AUTHORIZATION_TYPE))
      .filter(element -> isNotNoneValue().test(element.value.getLast()))
      .isPresent();
  }
  private static Predicate<Expression> isNotNoneValue() {
    return isFqn(AUTHORIZATION_TYPE_NONE).negate();
  }

  private static boolean isSafeAuthorisationArgument(SubscriptionContext subscriptionContext, CdkUtils.ExpressionFlow expression) {
    return expression.getExpression(isCallExpression().and(isFqn("aws_cdk.aws_apigateway.MethodOptions")))
      .flatMap(expr -> getArgument(subscriptionContext, (CallExpression) expr, AUTHORIZATION_TYPE))
      .filter(expr -> expr.hasExpression(isFqn(AUTHORIZATION_TYPE_NONE)))
      .isEmpty();
  }

  public static Predicate<Expression> isCallExpression() {
    return expression -> expression.is(Tree.Kind.CALL_EXPR);
  }
}
