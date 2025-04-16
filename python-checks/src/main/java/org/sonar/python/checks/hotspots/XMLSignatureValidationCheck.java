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

import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.DictionaryLiteral;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionList;
import org.sonar.plugins.python.api.tree.KeyValuePair;
import org.sonar.plugins.python.api.tree.ListLiteral;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.UnpackingExpression;
import org.sonar.plugins.python.api.types.v2.TriBool;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.TypeCheckBuilder;
import org.sonar.python.types.v2.TypeCheckMap;

import static org.sonar.python.checks.hotspots.CommonValidationUtils.singleAssignedString;

@Rule(key = "S6377")
public class XMLSignatureValidationCheck extends PythonSubscriptionCheck {

  private static final List<String> VERIFY_REQUIRED_ONE_OF = List.of(
    "x509_cert",
    "cert_subject_name",
    "cert_resolver",
    "ca_pem_file",
    "ca_path",
    "hmac_key"
  );
  private static final String MESSAGE = "Change this code to only accept signatures computed from a trusted party.";
  private static final String MESSAGE_SECONDARY = "Unsafe parameter set here";

  private TypeCheckBuilder xmlVerifierVerifyTypeChecker;
  private TypeCheckBuilder dictTypeChecker;
  private TypeCheckBuilder signatureConfigurationTypeChecker;
  private TypeCheckMap<Boolean> signatureMethodTypeCheckMap;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, this::registerTypeCheckers);
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, this::checkCallExpr);
  }

  private void registerTypeCheckers(SubscriptionContext subscriptionContext) {
    xmlVerifierVerifyTypeChecker = subscriptionContext.typeChecker().typeCheckBuilder().isTypeWithFqn("signxml.XMLVerifier");
    dictTypeChecker = subscriptionContext.typeChecker().typeCheckBuilder().isTypeWithName("dict");
    signatureConfigurationTypeChecker = subscriptionContext.typeChecker().typeCheckBuilder().isTypeWithFqn("signxml.SignatureConfiguration");
    signatureMethodTypeCheckMap = TypeCheckMap.ofEntries(
      Map.entry(subscriptionContext.typeChecker().typeCheckBuilder().isTypeWithFqn("signxml.SignatureMethod.HMAC_SHA224"), true),
      Map.entry(subscriptionContext.typeChecker().typeCheckBuilder().isTypeWithFqn("signxml.SignatureMethod.HMAC_SHA256"), true),
      Map.entry(subscriptionContext.typeChecker().typeCheckBuilder().isTypeWithFqn("signxml.SignatureMethod.HMAC_SHA384"), true),
      Map.entry(subscriptionContext.typeChecker().typeCheckBuilder().isTypeWithFqn("signxml.SignatureMethod.HMAC_SHA512"), true)
    );
  }

  private void checkCallExpr(SubscriptionContext subscriptionContext) {
    var callExpression = ((CallExpression) subscriptionContext.syntaxNode());
    var qualifiedExprCallee = Optional.of(callExpression)
      .map(CallExpression::callee)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(QualifiedExpression.class))
      .orElse(null);
    if (qualifiedExprCallee == null) {
      return;
    }
    var ok = qualifiedExpressionIsVerifyCall(qualifiedExprCallee);
    if (!ok) {
      return;
    }
    Map<String, Tree> argToTree = new HashMap<>();
    for (var arg : callExpression.arguments()) {
      if (arg instanceof RegularArgument regularArgument) {
        String argumentName = Optional.ofNullable(regularArgument.keywordArgument()).map(Name::name).orElse("");
        argToTree.put(argumentName, regularArgument.expression());
      } else {
        var keys = TreeUtils.toOptionalInstanceOf(UnpackingExpression.class, arg)
          .map(UnpackingExpression::expression)
          .flatMap(TreeUtils.toOptionalInstanceOfMapper(Name.class))
          .flatMap(Expressions::singleAssignedNonNameValue)
          .map(this::keysInUnpacking)
          .orElseGet(Map::of);
        argToTree.putAll(keys);
      }
    }

    if (Collections.disjoint(argToTree.keySet(), VERIFY_REQUIRED_ONE_OF)) {
      subscriptionContext.addIssue(callExpression.callee(), MESSAGE);
    }

    var expectConfigValue = argToTree.get("expect_config");
    if (expectConfigValue != null) {
      checkExpectConfig(subscriptionContext, expectConfigValue, callExpression);
    }
  }

  private void checkExpectConfig(SubscriptionContext subscriptionContext, Tree expectConfigValue, CallExpression verifyCallExpression) {
    expectConfigValue = replaceBySingleAssigned(expectConfigValue);
    if (expectConfigValue instanceof CallExpression callExpression
      && signatureConfigurationTypeChecker.check(callExpression.callee().typeV2()) == TriBool.TRUE) {
      TreeUtils.nthArgumentOrKeywordOptional(0, "require_x509", callExpression.arguments())
        .filter(arg -> Expressions.isFalsy(arg.expression()))
        .ifPresent(arg -> subscriptionContext.addIssue(verifyCallExpression, MESSAGE).secondary(arg, MESSAGE_SECONDARY));

      TreeUtils.nthArgumentOrKeywordOptional(3, "signature_methods", callExpression.arguments())
        .map(this::getOffendingInList)
        .filter(list -> !list.isEmpty())
        .ifPresent(list -> {
          var issue = subscriptionContext.addIssue(verifyCallExpression, MESSAGE);
          list.forEach(secondary -> issue.secondary(secondary, MESSAGE_SECONDARY));
        });
    }
  }

  private static Tree replaceBySingleAssigned(Tree expectConfigValue) {
    if (expectConfigValue.is(Tree.Kind.NAME)) {
      expectConfigValue = Expressions.singleAssignedNonNameValue((Name) expectConfigValue).orElse(null);
    }
    return expectConfigValue;
  }

  private List<Expression> getOffendingInList(RegularArgument regularArgument) {
    var expression = replaceBySingleAssigned(regularArgument.expression());
    return TreeUtils.toOptionalInstanceOf(ListLiteral.class, expression)
      .map(ListLiteral::elements)
      .map(ExpressionList::expressions)
      .stream()
      .flatMap(Collection::stream)
      .filter(e -> signatureMethodTypeCheckMap.getOptionalForType(e.typeV2()).isEmpty())
      .toList();
  }


  private Map<String, Tree> keysInUnpacking(Expression expression) {
    if (expression instanceof CallExpression callExpression && dictTypeChecker.check(callExpression.callee().typeV2()) == TriBool.TRUE) {
      return callExpression.arguments().stream()
        .flatMap(TreeUtils.toStreamInstanceOfMapper(RegularArgument.class))
        .filter(regularArgument -> regularArgument.keywordArgument() != null)
        .collect(Collectors.toMap(
          regularArgument -> regularArgument.keywordArgument().name(),
          RegularArgument::expression
        ));
    }
    if (expression instanceof DictionaryLiteral dictionaryLiteral) {
      var output = new HashMap<String, Tree>();
      for (var element : dictionaryLiteral.elements()) {
        if (element instanceof KeyValuePair keyValuePair) {

          var key = singleAssignedString(keyValuePair.key());
          if (!key.isEmpty()) {
            output.put(key, keyValuePair.value());
          }
        } else {
          var unpackedKeys = TreeUtils.toOptionalInstanceOf(UnpackingExpression.class, element)
            .map(UnpackingExpression::expression)
            .flatMap(TreeUtils.toOptionalInstanceOfMapper(Name.class))
            .flatMap(Expressions::singleAssignedNonNameValue)
            .map(this::keysInUnpacking)
            .orElseGet(Map::of);
          output.putAll(unpackedKeys);
        }
      }
      return output;
    }
    return Map.of();
  }

  private boolean qualifiedExpressionIsVerifyCall(QualifiedExpression qualifiedExprCallee) {
    if (qualifiedExprCallee.qualifier() instanceof CallExpression callExpression) {
      return xmlVerifierVerifyTypeChecker.check(callExpression.callee().typeV2()) == TriBool.TRUE;
    }
    if (qualifiedExprCallee.qualifier() instanceof Name name) {
      // This check can never be true because we don't have stubs for signxml.
      // Currently, if we try to instantiate the XMLVerifier and then call the verify method on the variable,
      // the name will have an unknown type.
      return xmlVerifierVerifyTypeChecker.check(name.typeV2()) == TriBool.TRUE;
    }
    return false;
  }
}
