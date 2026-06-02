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
package org.sonar.python.checks.cdk;

import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.Set;
import javax.annotation.CheckForNull;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.utils.Expressions;

import static org.sonar.python.checks.cdk.CdkPredicate.isFqn;
import static org.sonar.python.checks.cdk.CdkPredicate.isString;
import static org.sonar.python.checks.cdk.CdkPredicate.isStringLiteral;
import static org.sonar.python.checks.cdk.CdkUtils.getArgument;

@Rule(key = "S6333")
public class PublicApiIsSecuritySensitiveCheck extends AbstractCdkResourceCheck {

  private static final String MESSAGE = "Ensure this API route requires authentication.";
  private static final String AUTHORIZATION_TYPE = "authorization_type";
  private static final String ENDPOINT_CONFIGURATION = "endpoint_configuration";
  private static final String TYPES = "types";
  private static final String HTTP_METHOD = "http_method";
  private static final String PATH_PART = "path_part";
  private static final String AUTHORIZATION_TYPE_NONE = "aws_cdk.aws_apigateway.AuthorizationType.NONE";
  private static final String ADD_RESOURCE_FQN = "aws_cdk.aws_apigateway.Resource.add_resource";
  private static final String REST_API_FQN = "aws_cdk.aws_apigateway.RestApi";
  private static final String ENDPOINT_CONFIGURATION_FQN = "aws_cdk.aws_apigateway.EndpointConfiguration";
  private static final String ENDPOINT_TYPE_PRIVATE_FQN = "aws_cdk.aws_apigateway.EndpointType.PRIVATE";

  // Known public endpoints that do not require authentication by design.
  private static final Set<String> SAFE_PATH_NAMES = Set.of(
    "login", "signup", "register", "authenticate", "token",
    "forgot-password", "healthcheck", "health-check", "status",
    "callback", "public-keys", "jwks", "well-known"
  );

  // Route names that are strong indicators of sensitive, access-controlled areas.
  // Exact segment match only — "admin-portal" does not match "admin".
  private static final Set<String> SENSITIVE_PATH_NAMES = Set.of(
    "admin", "management", "internal"
  );

  private static final Set<String> DANGEROUS_HTTP_METHODS = Set.of(
    "POST", "PUT", "DELETE", "PATCH", "ANY"
  );

  @Override
  protected void registerFqnConsumer() {
    // L1 constructs: flag only when authorization_type is explicitly the literal "NONE".
    checkFqns(List.of("aws_cdk.aws_apigateway.CfnMethod", "aws_cdk.aws_apigatewayv2.CfnRoute"),
      (ctx, call) -> getArgument(ctx, call, AUTHORIZATION_TYPE)
        .ifPresent(arg -> arg.addIssueIf(isString("NONE"), MESSAGE))
    );

    // L2 construct: apply full detection pipeline.
    checkFqn("aws_cdk.aws_apigateway.Resource.add_method", PublicApiIsSecuritySensitiveCheck::checkAddMethod);
  }

  private static void checkAddMethod(SubscriptionContext ctx, CallExpression call) {
    // Must be explicit literal NONE — omission is not flagged.
    Optional<CdkUtils.ExpressionFlow> authArg = getArgument(ctx, call, AUTHORIZATION_TYPE);
    if (authArg.isEmpty() || !authArg.get().hasExpression(isFqn(AUTHORIZATION_TYPE_NONE))) {
      return;
    }

    // Path is normalised: lowercase and leading dots/slashes stripped so ".well-known" matches "well-known".
    Optional<String> path = resolveAddResourcePath(ctx, call).map(PublicApiIsSecuritySensitiveCheck::normalizePath);
    if (path.isPresent() && SAFE_PATH_NAMES.contains(path.get())) {
      return;
    }

    // Suppress: parent RestApi has a PRIVATE endpoint — the API is VPC-only and unreachable from the internet.
    if (isPrivateRestApi(ctx, call)) {
      return;
    }

    // Gate 1: state-changing HTTP method — sufficient to confirm a vulnerability.
    Optional<CdkUtils.ExpressionFlow> httpMethod = getArgument(ctx, call, HTTP_METHOD, 0);
    if (httpMethod.isPresent() && httpMethod.get().hasExpression(isString(DANGEROUS_HTTP_METHODS))) {
      authArg.get().addIssue(MESSAGE);
      return;
    }

    if (path.isPresent() && SENSITIVE_PATH_NAMES.contains(path.get())) {
      authArg.get().addIssue(MESSAGE);
    }
  }

  private static boolean isPrivateRestApi(SubscriptionContext ctx, CallExpression addMethodCall) {
    Expression callee = addMethodCall.callee();
    if (!callee.is(Tree.Kind.QUALIFIED_EXPR)) {
      return false;
    }
    Expression qualifier = ((QualifiedExpression) callee).qualifier();
    CallExpression addResourceCall = resolveToAddResourceCall(qualifier);
    if (addResourceCall != null) {
      return walkChainToRestApi(addResourceCall)
        .map(restApiCall -> hasPrivateEndpointConfiguration(ctx, restApiCall))
        .orElse(false);
    }
    // Fallback: add_method called directly on api.root (no add_resource in chain)
    return resolveToRestApiFromRoot(qualifier)
        .map(restApiCall -> hasPrivateEndpointConfiguration(ctx, restApiCall))
        .orElse(false);
  }

  private static Optional<CallExpression> walkChainToRestApi(CallExpression addResourceCall) {
    Expression callee = addResourceCall.callee();
    if (!callee.is(Tree.Kind.QUALIFIED_EXPR)) {
      return Optional.empty();
    }
    Expression qualifier = ((QualifiedExpression) callee).qualifier();

    // api.root.add_resource(...) — qualifier is a QualifiedExpression ending in .root
    if (qualifier.is(Tree.Kind.QUALIFIED_EXPR)) {
      QualifiedExpression qe = (QualifiedExpression) qualifier;
      if ("root".equals(qe.name().name())) {
        return resolveToRestApiCall(qe.qualifier());
      }
    }

    // api.root.add_resource("a").add_resource("b") — qualifier is itself an add_resource call
    if (qualifier.is(Tree.Kind.CALL_EXPR)) {
      CallExpression inner = (CallExpression) qualifier;
      if (isAddResourceCall(inner)) {
        return walkChainToRestApi(inner);
      }
    }

    // resource = api.root.add_resource("a"); resource.add_resource("b")
    if (qualifier.is(Tree.Kind.NAME)) {
      Expression assigned = Expressions.singleAssignedValue((Name) qualifier);
      if (assigned != null && assigned.is(Tree.Kind.CALL_EXPR)) {
        CallExpression inner = (CallExpression) assigned;
        if (isAddResourceCall(inner)) {
          return walkChainToRestApi(inner);
        }
      }
    }

    return Optional.empty();
  }

  private static Optional<CallExpression> resolveToRestApiCall(Expression expression) {
    Expression resolved = expression;
    if (expression.is(Tree.Kind.NAME)) {
      Expression assigned = Expressions.singleAssignedValue((Name) expression);
      if (assigned != null) {
        resolved = assigned;
      }
    }
    if (resolved.is(Tree.Kind.CALL_EXPR)) {
      CallExpression call = (CallExpression) resolved;
      if (Optional.ofNullable(call.calleeSymbol())
        .map(Symbol::fullyQualifiedName)
        .filter(REST_API_FQN::equals)
        .isPresent()) {
        return Optional.of(call);
      }
    }
    return Optional.empty();
  }

  private static Optional<CallExpression> resolveToRestApiFromRoot(Expression expression) {
    Expression resolved = expression;
    if (expression.is(Tree.Kind.NAME)) {
      Expression assigned = Expressions.singleAssignedValue((Name) expression);
      if (assigned != null) {
        resolved = assigned;
      }
    }
    if (resolved.is(Tree.Kind.QUALIFIED_EXPR)) {
      QualifiedExpression qe = (QualifiedExpression) resolved;
      if ("root".equals(qe.name().name())) {
        return resolveToRestApiCall(qe.qualifier());
      }
    }
    return Optional.empty();
  }

  private static boolean hasPrivateEndpointConfiguration(SubscriptionContext ctx, CallExpression restApiCall) {
    return getArgument(ctx, restApiCall, ENDPOINT_CONFIGURATION)
      .flatMap(flow -> flow.getExpression(e -> e.is(Tree.Kind.CALL_EXPR) && isFqn(ENDPOINT_CONFIGURATION_FQN).test(e)))
      .map(CallExpression.class::cast)
      .flatMap(configCall -> getArgument(ctx, configCall, TYPES, 0))
      .flatMap(CdkUtils::getList)
      .map(list -> CdkUtils.getListElements(ctx, list))
      .map(elements -> elements.stream().anyMatch(el -> el.hasExpression(isFqn(ENDPOINT_TYPE_PRIVATE_FQN))))
      .orElse(false);
  }

  private static Optional<String> resolveAddResourcePath(SubscriptionContext ctx, CallExpression addMethodCall) {
    Expression callee = addMethodCall.callee();
    if (!callee.is(Tree.Kind.QUALIFIED_EXPR)) {
      return Optional.empty();
    }
    Expression qualifier = ((QualifiedExpression) callee).qualifier();
    CallExpression addResourceCall = resolveToAddResourceCall(qualifier);
    if (addResourceCall == null) {
      return Optional.empty();
    }
    return getArgument(ctx, addResourceCall, PATH_PART, 0)
      .flatMap(flow -> flow.getExpression(isStringLiteral()))
      .flatMap(CdkUtils::getString);
  }

  @CheckForNull
  private static CallExpression resolveToAddResourceCall(Expression expression) {
    if (expression.is(Tree.Kind.CALL_EXPR)) {
      CallExpression call = (CallExpression) expression;
      return isAddResourceCall(call) ? call : null;
    }
    if (expression.is(Tree.Kind.NAME)) {
      Expression assigned = Expressions.singleAssignedValue((Name) expression);
      if (assigned != null && assigned.is(Tree.Kind.CALL_EXPR)) {
        CallExpression call = (CallExpression) assigned;
        return isAddResourceCall(call) ? call : null;
      }
    }
    return null;
  }

  private static String normalizePath(String path) {
    return path.replaceAll("^[./]+", "").toLowerCase(Locale.ROOT);
  }

  private static boolean isAddResourceCall(CallExpression call) {
    return Optional.ofNullable(call.calleeSymbol())
      .map(Symbol::fullyQualifiedName)
      .filter(ADD_RESOURCE_FQN::equals)
      .isPresent();
  }
}
