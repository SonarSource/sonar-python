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
package org.sonar.python.checks;

import java.util.Map;
import java.util.Optional;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.SubscriptionExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TypeAnnotation;
import org.sonar.python.quickfix.TextEditUtils;

import static java.util.Map.entry;

@Rule(key = "S6545")
public class BuiltinGenericsOverTypingModuleCheck extends PythonSubscriptionCheck {

  public static final String MESSAGE = "Use the built-in generic type `%s` instead of its typing counterpart.";
  private static final Map<String, String> GENERICS_NAME = Map.ofEntries(
    entry("typing.List", "list"),
    entry("typing.Dict", "dict"),
    entry("typing.Tuple", "tuple"),
    entry("typing.Set", "set"),
    entry("typing.FrozenSet", "frozenset"),
    entry("typing.Type", "type"),
    entry("typing.Iterable", "collections.abc.Iterable"),
    entry("typing.AbstractSet", "collections.abc.Set"),
    entry("typing.Callable", "collections.abc.Callable"),
    entry("typing.Mapping", "collections.abc.Mapping"),
    entry("typing.Sequence", "collections.abc.Sequence"),
    entry("typing.deque", "collections.deque"),
    entry("typing.defaultdict", "collections.defaultdict"),
    entry("typing.OrderedDict", "collections.OrderedDict"),
    entry("typing.Counter", "collections.Counter"),
    entry("typing.ChainMap", "collections.ChainMap"));

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.RETURN_TYPE_ANNOTATION, BuiltinGenericsOverTypingModuleCheck::checkForTypingModule);
    context.registerSyntaxNodeConsumer(Tree.Kind.PARAMETER_TYPE_ANNOTATION, BuiltinGenericsOverTypingModuleCheck::checkForTypingModule);
    context.registerSyntaxNodeConsumer(Tree.Kind.VARIABLE_TYPE_ANNOTATION, BuiltinGenericsOverTypingModuleCheck::checkForTypingModule);
  }

  private static void checkForTypingModule(SubscriptionContext subscriptionContext) {
    TypeAnnotation typeAnnotation = (TypeAnnotation) subscriptionContext.syntaxNode();
    Expression expression = typeAnnotation.expression();
    checkForGenericsFromTypingModule(subscriptionContext, expression);
  }

  private static void checkForGenericsFromTypingModule(SubscriptionContext subscriptionContext, Expression expression) {
    if (expression instanceof SubscriptionExpression subscriptionExpression) {
      getGenericsCounterPartFromTypingModule(subscriptionContext, subscriptionExpression)
        .ifPresent(preferredGenerics -> raiseIssueForGenerics(subscriptionContext, subscriptionExpression, preferredGenerics));
    }
  }

  private static void raiseIssueForGenerics(SubscriptionContext context, SubscriptionExpression expression, String preferredGenerics) {
    String specificMessage = String.format(MESSAGE, preferredGenerics);
    PreciseIssue preciseIssue = context.addIssue(expression, specificMessage);
    addQuickFix(preciseIssue, expression, preferredGenerics, specificMessage);
  }

  private static void addQuickFix(PreciseIssue issue, SubscriptionExpression expression, String preferredGenerics, String message) {
    // Ignoring quick fix if the change would require an import
    if (!preferredGenerics.contains(".")) {
      PythonQuickFix quickFix = PythonQuickFix.newQuickFix(message)
        .addTextEdit(
          TextEditUtils.replaceRange(expression.firstToken(), expression.leftBracket(), preferredGenerics + "["))
        .build();
      issue.addQuickFix(quickFix);
    }
  }

  private static Optional<String> getGenericsCounterPartFromTypingModule(SubscriptionContext context, SubscriptionExpression expression) {
    // Recursive check on nested types
    expression.subscripts().expressions()
      .forEach(nestedExpression -> checkForGenericsFromTypingModule(context, nestedExpression));
    return Optional.of(expression.object())
      .map(HasSymbol.class::cast)
      .map(HasSymbol::symbol)
      .flatMap(BuiltinGenericsOverTypingModuleCheck::getBuiltinGenericsType);
  }

  private static Optional<String> getBuiltinGenericsType(@Nullable Symbol maybeSymbol) {
    return Optional.ofNullable(maybeSymbol)
      .map(Symbol::fullyQualifiedName)
      .flatMap(fullyQualifiedName -> Optional.ofNullable(GENERICS_NAME.get(fullyQualifiedName)));
  }

}
