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
package org.sonar.python.checks;

import java.util.Map;
import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.symbols.v2.SymbolV2;
import org.sonar.plugins.python.api.symbols.v2.UsageV2;
import org.sonar.plugins.python.api.tree.AliasedName;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ImportFrom;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.SubscriptionExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TypeAnnotation;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.tree.TreeUtils;

import static java.util.Map.entry;

@Rule(key = "S6545")
public class BuiltinGenericsOverTypingModuleCheck extends PythonSubscriptionCheck {

  public static final String MESSAGE = "Use the built-in generic type `%s` instead of its typing counterpart.";
  private static final TypeMatcher TYPING_MODULE = TypeMatchers.isType("typing");
  private static final Map<String, String> GENERIC_TYPE_REPLACEMENTS = Map.ofEntries(
    entry("List", "list"),
    entry("Dict", "dict"),
    entry("Tuple", "tuple"),
    entry("Set", "set"),
    entry("FrozenSet", "frozenset"),
    entry("Type", "type"),
    entry("Iterable", "collections.abc.Iterable"),
    entry("AbstractSet", "collections.abc.Set"),
    entry("Callable", "collections.abc.Callable"),
    entry("Mapping", "collections.abc.Mapping"),
    entry("Sequence", "collections.abc.Sequence"),
    entry("deque", "collections.deque"),
    entry("defaultdict", "collections.defaultdict"),
    entry("OrderedDict", "collections.OrderedDict"),
    entry("Counter", "collections.Counter"),
    entry("ChainMap", "collections.ChainMap"));

  private boolean hasWildcardTypingImport;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT,
      ctx -> hasWildcardTypingImport = TreeUtils.hasDescendant(ctx.syntaxNode(), BuiltinGenericsOverTypingModuleCheck::isWildcardTypingImport));
    context.registerSyntaxNodeConsumer(Tree.Kind.RETURN_TYPE_ANNOTATION, this::checkForTypingModule);
    context.registerSyntaxNodeConsumer(Tree.Kind.PARAMETER_TYPE_ANNOTATION, this::checkForTypingModule);
    context.registerSyntaxNodeConsumer(Tree.Kind.VARIABLE_TYPE_ANNOTATION, this::checkForTypingModule);
  }

  private void checkForTypingModule(SubscriptionContext subscriptionContext) {
    TypeAnnotation typeAnnotation = (TypeAnnotation) subscriptionContext.syntaxNode();
    Expression expression = typeAnnotation.expression();
    checkForGenericsFromTypingModule(subscriptionContext, expression);
  }

  private void checkForGenericsFromTypingModule(SubscriptionContext subscriptionContext, Expression expression) {
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

  private Optional<String> getGenericsCounterPartFromTypingModule(SubscriptionContext context, SubscriptionExpression expression) {
    // Recursive check on nested types
    expression.subscripts().expressions()
      .forEach(nestedExpression -> checkForGenericsFromTypingModule(context, nestedExpression));
    Expression object = expression.object();
    if (object instanceof QualifiedExpression qualifiedExpression
      && TYPING_MODULE.isTrueFor(qualifiedExpression.qualifier(), context)) {
      return Optional.ofNullable(GENERIC_TYPE_REPLACEMENTS.get(qualifiedExpression.name().name()));
    }
    if (object instanceof Name name) {
      return getImportedTypingGeneric(name);
    }
    return Optional.empty();
  }

  private Optional<String> getImportedTypingGeneric(Name name) {
    SymbolV2 symbol = name.symbolV2();
    Optional<UsageV2> bindingUsage = Optional.ofNullable(symbol)
      .flatMap(value -> value.usages().stream()
        .filter(UsageV2::isBindingUsage)
        .reduce((first, second) -> second));
    return bindingUsage
      .map(UsageV2::tree)
      .flatMap(BuiltinGenericsOverTypingModuleCheck::getImportedNameFromTyping)
      .or(() -> symbol == null || symbol.usages().stream().noneMatch(UsageV2::isBindingUsage)
        ? getGenericFromWildcardTypingImport(name)
        : Optional.empty());
  }

  private Optional<String> getGenericFromWildcardTypingImport(Name name) {
    return hasWildcardTypingImport
      ? Optional.ofNullable(GENERIC_TYPE_REPLACEMENTS.get(name.name()))
      : Optional.empty();
  }

  private static Optional<String> getImportedNameFromTyping(Tree bindingTree) {
    Tree importTree = TreeUtils.firstAncestor(bindingTree, ImportFrom.class::isInstance);
    if (!(importTree instanceof ImportFrom importFrom) || !isTypingImport(importFrom)) {
      return Optional.empty();
    }
    return Optional.ofNullable(GENERIC_TYPE_REPLACEMENTS.get(nameFromImport(bindingTree)));
  }

  private static boolean isWildcardTypingImport(Tree tree) {
    return tree instanceof ImportFrom importFrom && importFrom.isWildcardImport() && isTypingImport(importFrom);
  }

  private static boolean isTypingImport(ImportFrom importFrom) {
    return importFrom.module() != null
      && importFrom.dottedPrefixForModule().isEmpty()
      && importFrom.module().names().size() == 1
      && "typing".equals(importFrom.module().names().getFirst().name());
  }

  private static String nameFromImport(Tree bindingTree) {
    Tree current = bindingTree;
    while (!(current instanceof AliasedName)) {
      current = current.parent();
    }
    return ((AliasedName) current).dottedName().names().getLast().name();
  }

}
