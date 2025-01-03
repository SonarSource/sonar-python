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
package org.sonar.python.checks;

import java.util.Collection;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Stream;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.DictionaryLiteral;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionList;
import org.sonar.plugins.python.api.tree.KeyValuePair;
import org.sonar.plugins.python.api.tree.ListLiteral;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.SubscriptionExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.plugins.python.api.tree.Tuple;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S5659")
public class JwtVerificationCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Don't use a JWT token without verifying its signature.";

  // https://github.com/davedoesdev/python-jwt
  // "From version 2.0.1 the namespace has changed from jwt to python_jwt, in order to avoid conflict with PyJWT"
  private static final Set<String> PROCESS_JWT_FQNS = Set.of(
    "python_jwt.process_jwt",
    "jwt.process_jwt");

  private static final Set<String> VERIFY_JWT_FQNS = Set.of(
    "python_jwt.verify_jwt",
    "jwt.verify_jwt");

  private static final Set<String> ALLOWED_KEYS_ACCESS = Set.of("jku", "jwk", "kid", "x5u", "x5c", "x5t", "xt#256");

  private static final Set<String> WHERE_VERIFY_KWARG_SHOULD_BE_TRUE_FQNS = Set.of(
    "jwt.decode",
    "jose.jws.verify");

  private static final Set<String> UNVERIFIED_FQNS = Set.of(
    "jwt.get_unverified_header",
    "jose.jwt.get_unverified_header",
    "jose.jwt.get_unverified_headers",
    "jose.jws.get_unverified_header",
    "jose.jws.get_unverified_headers",
    "jose.jwt.get_unverified_claims",
    "jose.jws.get_unverified_claims");

  private static final String VERIFY_SIGNATURE_KEYWORD = "verify_signature";

  public static final Set<String> VERIFY_SIGNATURE_OPTION_SUPPORTING_FUNCTION_FQNS = Set.of("jose.jwt.decode", "jwt.decode");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Kind.CALL_EXPR, JwtVerificationCheck::verifyCallExpression);
  }

  private static void verifyCallExpression(SubscriptionContext ctx) {
    CallExpression call = (CallExpression) ctx.syntaxNode();

    Symbol calleeSymbol = call.calleeSymbol();
    if (calleeSymbol == null || calleeSymbol.fullyQualifiedName() == null) {
      return;
    }

    String calleeFqn = calleeSymbol.fullyQualifiedName();
    if (WHERE_VERIFY_KWARG_SHOULD_BE_TRUE_FQNS.contains(calleeFqn)) {
      RegularArgument verifyArg = TreeUtils.argumentByKeyword("verify", call.arguments());
      if (verifyArg != null && Expressions.isFalsy(verifyArg.expression())) {
        ctx.addIssue(verifyArg, MESSAGE);
        return;
      }
    } else if (PROCESS_JWT_FQNS.contains(calleeFqn)) {
      Optional.ofNullable(TreeUtils.firstAncestorOfKind(call, Kind.FILE_INPUT, Kind.FUNCDEF))
        .filter(scriptOrFunction -> !TreeUtils.hasDescendant(scriptOrFunction, JwtVerificationCheck::isCallToVerifyJwt))
        .ifPresent(scriptOrFunction -> ctx.addIssue(call, MESSAGE));
    } else if (UNVERIFIED_FQNS.contains(calleeFqn) && !accessOnlyAllowedHeaderKeys(call)) {
      Optional.ofNullable(TreeUtils.nthArgumentOrKeyword(0, "", call.arguments()))
        .flatMap(TreeUtils.toOptionalInstanceOfMapper(RegularArgument.class))
        .map(RegularArgument::expression)
        .ifPresent(argument -> ctx.addIssue(argument, MESSAGE));
    }
    if (VERIFY_SIGNATURE_OPTION_SUPPORTING_FUNCTION_FQNS.contains(calleeFqn)) {
      Optional.ofNullable(TreeUtils.argumentByKeyword("options", call.arguments()))
        .map(RegularArgument::expression)
        .filter(JwtVerificationCheck::isListOrDictWithSensitiveEntry)
        .ifPresent(expression -> ctx.addIssue(expression, MESSAGE));
    }
  }

  private static boolean isListOrDictWithSensitiveEntry(@Nullable Expression expression) {
    if (expression == null) {
      return false;
    } else if (expression.is(Tree.Kind.NAME)) {
      return isListOrDictWithSensitiveEntry(Expressions.singleAssignedNonNameValue((Name) expression).orElse(null));
    } else if (expression.is(Kind.DICTIONARY_LITERAL)) {
      return hasTrueVerifySignatureEntry((DictionaryLiteral) expression);
    } else if (expression.is(Kind.LIST_LITERAL)) {
      return hasTrueVerifySignatureEntry((ListLiteral) expression);
    } else if (expression.is(Kind.CALL_EXPR)) {
      return isCallToDict((CallExpression) expression)
        && hasIllegalDictKWArgument((CallExpression) expression);
    }
    return false;
  }

  private static boolean hasIllegalDictKWArgument(CallExpression expression) {
    return Optional.of(expression)
      .map(CallExpression::arguments)
      .map(arguments -> TreeUtils.argumentByKeyword(VERIFY_SIGNATURE_KEYWORD, arguments))
      .map((RegularArgument::expression))
      .filter(Expressions::isFalsy)
      .isPresent();
  }

  private static boolean isCallToDict(CallExpression expression) {
    return Optional.of(expression)
      .map(CallExpression::calleeSymbol)
      .map(Symbol::fullyQualifiedName)
      .filter("dict"::equals).isPresent();
  }

  private static boolean hasTrueVerifySignatureEntry(DictionaryLiteral dictionaryLiteral) {
    return dictionaryLiteral.elements().stream()
      .filter(KeyValuePair.class::isInstance)
      .map(KeyValuePair.class::cast)
      .filter(keyValuePair -> isSensitiveKey(keyValuePair.key()))
      .map(KeyValuePair::value)
      .anyMatch(Expressions::isFalsy);
  }

  private static boolean hasTrueVerifySignatureEntry(ListLiteral listLiteral) {
    return listLiteral.elements().expressions().stream()
      .filter(Tuple.class::isInstance)
      .map(Tuple.class::cast)
      .map(Tuple::elements)
      .filter(list -> list.size() == 2)
      .filter(list -> isSensitiveKey(list.get(0)))
      .map(list -> list.get(1))
      .anyMatch(Expressions::isFalsy);
  }

  private static boolean isSensitiveKey(Expression key) {
    return key.is(Kind.STRING_LITERAL) && VERIFY_SIGNATURE_KEYWORD.equals(((StringLiteral) key).trimmedQuotesValue());
  }

  private static boolean isCallToVerifyJwt(Tree t) {
    return TreeUtils.toOptionalInstanceOf(CallExpression.class, t)
      .map(CallExpression::calleeSymbol)
      .map(Symbol::fullyQualifiedName)
      .filter(VERIFY_JWT_FQNS::contains)
      .isPresent();
  }

  private static boolean accessOnlyAllowedHeaderKeys(CallExpression call) {
    Tree assignment = TreeUtils.firstAncestorOfKind(call, Tree.Kind.ASSIGNMENT_STMT);
    Stream<StringLiteral> headerKeysAccessedDirectly = accessToHeaderKeyWithoutAssignment(call);
    if (assignment == null) {
      return areStringLiteralsPartOfAllowedKeys(headerKeysAccessedDirectly);
    } else {
      List<Expression> lhsExpressions = ((AssignmentStatement) assignment).lhsExpressions().stream()
        .map(ExpressionList::expressions)
        .flatMap(Collection::stream).toList();
      if (lhsExpressions.size() == 1 && lhsExpressions.get(0).is(Tree.Kind.NAME)) {
        Name name = (Name) lhsExpressions.get(0);
        Symbol symbol = name.symbol();
        if (symbol != null) {
          Stream<StringLiteral> argumentsOfGet = usagesAccessedByGet(symbol, call);
          Stream<StringLiteral> argumentsOfSubscription = usagesAccessedBySubscription(symbol, call);
          Stream<StringLiteral> headerKeysAccessFromAssignedValues = Stream.concat(argumentsOfGet, argumentsOfSubscription);
          return areStringLiteralsPartOfAllowedKeys(Stream.concat(headerKeysAccessFromAssignedValues, headerKeysAccessedDirectly));
        }
      }
    }
    return false;
  }

  private static boolean areStringLiteralsPartOfAllowedKeys(Stream<StringLiteral> literals) {
    var literalList = literals.toList();
    return !literalList.isEmpty() && literalList.stream().allMatch(str -> ALLOWED_KEYS_ACCESS.contains(str.trimmedQuotesValue()));
  }

  private static Stream<StringLiteral> accessToHeaderKeyWithoutAssignment(CallExpression call) {
    Stream<CallExpression> callExpressionFromGetUnverifiedHeaders = getCallExprWhereDictIsAccessedWithGet(Stream.of(call.parent()));
    Stream<Argument> argumentsOfCallExpr = getArgumentsFromCallExpr(callExpressionFromGetUnverifiedHeaders);
    Stream<StringLiteral> stringLiteralArgumentsFromGet = getStringLiteralArguments(argumentsOfCallExpr);
    Stream<SubscriptionExpression> subscriptionFromGetUnverifiedHeaders = getSubscriptions(Stream.of(call.parent()));
    Stream<StringLiteral> stringLiteralArgumentFromSubscription = getSubscriptsStringLiteral(subscriptionFromGetUnverifiedHeaders);
    return Stream.concat(stringLiteralArgumentsFromGet, stringLiteralArgumentFromSubscription);
  }

  private static Stream<StringLiteral> usagesAccessedByGet(Symbol symbol, CallExpression call) {
    var usages = getForwardUsages(symbol, call);
    var parentOfUsages = usages.map(Usage::tree).map(Tree::parent);
    var callExpressionsFromUsages = getCallExprWhereDictIsAccessedWithGet(parentOfUsages);
    return getStringLiteralArguments(getArgumentsFromCallExpr(callExpressionsFromUsages));
  }

  private static Stream<Argument> getArgumentsFromCallExpr(Stream<CallExpression> callExprs) {
    return callExprs.map(CallExpression::arguments).flatMap(Collection::stream);
  }

  private static Stream<Usage> getForwardUsages(Symbol symbol, CallExpression call) {
    return symbol.usages().stream()
      .filter(usage -> usage.tree().firstToken().line() > call.callee().firstToken().line());
  }

  private static Stream<CallExpression> getCallExprWhereDictIsAccessedWithGet(Stream<Tree> parentQualifiedExpr) {
    return parentQualifiedExpr
      .filter(parent -> parent.is(Tree.Kind.QUALIFIED_EXPR))
      .flatMap(TreeUtils.toStreamInstanceOfMapper(QualifiedExpression.class))
      .filter(expr -> "get".equals(expr.name().name()))
      .filter(expr -> expr.parent().is(Kind.CALL_EXPR))
      .map(QualifiedExpression::parent)
      .flatMap(TreeUtils.toStreamInstanceOfMapper(CallExpression.class));
  }

  private static Stream<StringLiteral> getStringLiteralArguments(Stream<Argument> arguments) {
    return arguments.filter(arg -> arg.is(Tree.Kind.REGULAR_ARGUMENT))
      .flatMap(TreeUtils.toStreamInstanceOfMapper(RegularArgument.class))
      .map(RegularArgument::expression)
      .flatMap(TreeUtils.toStreamInstanceOfMapper(StringLiteral.class));
  }

  private static Stream<StringLiteral> usagesAccessedBySubscription(Symbol symbol, CallExpression call) {
    var usages = getForwardUsages(symbol, call);
    var parentFromUsages = usages.map(Usage::tree).map(Tree::parent);
    var subscriptionsFromUsages = getSubscriptions(parentFromUsages);
    return getSubscriptsStringLiteral(subscriptionsFromUsages);
  }

  private static Stream<SubscriptionExpression> getSubscriptions(Stream<Tree> subscriptions) {
    return subscriptions
      .filter(subscription -> subscription.is(Tree.Kind.SUBSCRIPTION))
      .flatMap(TreeUtils.toStreamInstanceOfMapper(SubscriptionExpression.class));
  }

  private static Stream<StringLiteral> getSubscriptsStringLiteral(Stream<SubscriptionExpression> subscriptions) {
    return subscriptions.map(SubscriptionExpression::subscripts)
      .map(ExpressionList::expressions)
      .flatMap(Collection::stream)
      .flatMap(TreeUtils.toStreamInstanceOfMapper(StringLiteral.class));
  }

}
