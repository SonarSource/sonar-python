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

import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.function.Consumer;
import java.util.stream.Collectors;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.DictionaryLiteral;
import org.sonar.plugins.python.api.tree.DictionaryLiteralElement;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.KeyValuePair;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.StringElement;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.BuiltinTypes;
import org.sonar.python.checks.utils.Expressions;

public abstract class AbstractStringFormatCheck extends PythonSubscriptionCheck {

  protected static final Consumer<String> IGNORE_SYNTAX_ERRORS = message -> {};

  private static final List<String> NOT_MAPPING_TYPES = Arrays.asList(BuiltinTypes.LIST, BuiltinTypes.TUPLE, BuiltinTypes.STR);

  protected AbstractStringFormatCheck() {

  }

  protected void checkPrintfStyle(SubscriptionContext ctx) {
    BinaryExpression expression = (BinaryExpression) ctx.syntaxNode();
    StringLiteral literal = extractStringLiteral(expression.leftOperand());
    if (literal == null) {
      return;
    }

    if (literal.stringElements().stream().anyMatch(AbstractStringFormatCheck::isFStringOrBytesLiteral)) {
      // Do not bother with byte formatting and f-strings for now.
      return;
    }

    this.checkPrintfStyle(ctx, expression, literal);
  }

  protected abstract void checkPrintfStyle(SubscriptionContext ctx, BinaryExpression modulo, StringLiteral literal);

  protected static void checkPrintfDictionary(SubscriptionContext ctx, StringFormat format, DictionaryLiteral dict) {
    // Check the keys - do not bother with dictionaries containing unpacking expressions or keys which are not string literals
    for (DictionaryLiteralElement element : dict.elements()) {
      if (!element.is(Tree.Kind.KEY_VALUE_PAIR)) {
        return;
      }

      KeyValuePair pair = (KeyValuePair) element;
      if (!pair.key().type().canOnlyBe("str")) {
        ctx.addIssue(pair.key(), "Replace this key; %-format accepts only string keys.");
        return;
      }

      if (!pair.key().is(Tree.Kind.STRING_LITERAL)) {
        return;
      }
    }

    Map<String, List<StringFormat.ReplacementField>> fieldMap = format.replacementFields().stream()
      .collect(Collectors.groupingBy(StringFormat.ReplacementField::name));
    for (DictionaryLiteralElement element : dict.elements()) {
      KeyValuePair pair = (KeyValuePair) element;
      String key = ((StringLiteral) pair.key()).trimmedQuotesValue();

      List<StringFormat.ReplacementField> fields = fieldMap.remove(key);
      if (fields == null) {
        // No such field
        continue;
      }

      fields.forEach(field -> field.validateArgument(ctx, pair.value()));
    }

    // Check if we have any unmatched field names left
    fieldMap.keySet().forEach(fieldName -> ctx.addIssue(dict, String.format("Provide a value for field \"%s\".", fieldName)));
  }

  protected static void checkPrintfExpressionList(SubscriptionContext ctx, StringFormat format, Token locFrom, Token locTo, List<Expression> expressions) {
    if (format.numExpectedArguments() != expressions.size()) {
      reportInvalidArgumentSize(ctx, locFrom, locTo, format.numExpectedArguments(), expressions.size());
      return;
    }

    for (int i = 0; i < expressions.size(); ++i) {
      format.replacementFields().get(i).validateArgument(ctx, expressions.get(i));
    }
  }

  protected void checkStrFormatStyle(SubscriptionContext ctx) {
    CallExpression callExpression = (CallExpression) ctx.syntaxNode();
    if (!isQualifiedCallToStrFormat(callExpression)) {
      return;
    }

    Expression qualifier = ((QualifiedExpression) callExpression.callee()).qualifier();
    StringLiteral literal = extractStringLiteral(qualifier);
    if (literal == null) {
      return;
    }

    if (literal.stringElements().stream().anyMatch(AbstractStringFormatCheck::isFStringOrBytesLiteral)) {
      // Avoid raising on f-strings
      return;
    }

    this.checkStrFormatStyle(ctx, callExpression, qualifier, literal);
  }

  protected abstract void checkStrFormatStyle(SubscriptionContext ctx, CallExpression callExpression, Expression qualifier, StringLiteral literal);

  protected static boolean isQualifiedCallToStrFormat(CallExpression callExpression) {
    Symbol symbol = callExpression.calleeSymbol();
    return callExpression.callee().is(Tree.Kind.QUALIFIED_EXPR)
      && symbol != null
      && "str.format".equals(symbol.fullyQualifiedName());
  }

  protected static Consumer<String> syntaxIssueReporter(SubscriptionContext ctx, Tree primary, Tree secondary) {
    return message -> reportIssue(ctx, primary, secondary, message);
  }

  protected static void reportIssue(SubscriptionContext ctx, Tree primary, Tree secondary, String message) {
    PythonCheck.PreciseIssue preciseIssue = ctx.addIssue(primary, message);
    if (primary != secondary) {
      preciseIssue.secondary(secondary, null);
    }
  }

  protected static void reportInvalidArgumentSize(SubscriptionContext ctx, Token locFrom, Token locTo, long expected, long actual) {
    if (expected > actual) {
      ctx.addIssue(locFrom, locTo, String.format("Add %d missing argument(s).", expected - actual));
    } else {
      ctx.addIssue(locFrom, locTo, String.format("Remove %d unexpected argument(s).", actual - expected));
    }
  }

  protected static StringLiteral extractStringLiteral(Tree tree) {
    if (tree.is(Tree.Kind.STRING_LITERAL)) {
      return (StringLiteral) tree;
    }

    if (tree.is(Tree.Kind.NAME)) {
      Expression assignedValue = Expressions.singleAssignedValue(((Name) tree));
      if (assignedValue != null && assignedValue.is(Tree.Kind.STRING_LITERAL)) {
        return ((StringLiteral) assignedValue);
      }
    }

    return null;
  }

  protected static boolean isMapping(Expression expression) {
    // We consider everything having __getitem__ a mapping, with the exception of list and tuple.
    return NOT_MAPPING_TYPES.stream().noneMatch(type -> expression.type().canOnlyBe(type))
      && expression.type().canHaveMember("__getitem__");
  }

  private static boolean isFStringOrBytesLiteral(StringElement stringElement) {
    String prefix = stringElement.prefix().toLowerCase(Locale.ENGLISH);
    return prefix.contains("b") || prefix.contains("f");
  }

}
