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

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalInt;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import org.sonar.check.Rule;
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
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tuple;
import org.sonar.plugins.python.api.types.BuiltinTypes;

@Rule(key = "S2275")
public class StringFormatMisuseCheck extends PythonSubscriptionCheck {

  private static final List<String> NOT_MAPPING_TYPES = Arrays.asList("list", "tuple", "str");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.MODULO, StringFormatMisuseCheck::checkPrintfStyle);
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, StringFormatMisuseCheck::checkStrFormatStyle);
  }

  private static boolean isQualifiedCallToStrFormat(CallExpression callExpression) {
    Symbol symbol = callExpression.calleeSymbol();
    return callExpression.callee().is(Tree.Kind.QUALIFIED_EXPR)
      && symbol != null
      && "str.format".equals(symbol.fullyQualifiedName());
  }

  private static void checkStrFormatStyle(SubscriptionContext ctx) {
    CallExpression callExpression = (CallExpression) ctx.syntaxNode();
    if (!isQualifiedCallToStrFormat(callExpression)) {
      return;
    }

    Expression qualifier = ((QualifiedExpression) callExpression.callee()).qualifier();
    StringLiteral literal = extractStringLiteral(qualifier);
    if (literal == null) {
      return;
    }

    // Check the arguments for out of scope cases before we try to parse the string
    if (callExpression.arguments().stream().anyMatch(argument -> !argument.is(Tree.Kind.REGULAR_ARGUMENT))) {
      return;
    }

    Optional<StringFormat> format = StringFormat.createFromStrFormatStyle(ctx, qualifier, literal);
    if (!format.isPresent()) {
      return;
    }

    List<RegularArgument> arguments = callExpression.arguments().stream()
      .map(RegularArgument.class::cast)
      .collect(Collectors.toList());

    OptionalInt firstKwIdx = IntStream.range(0, arguments.size())
      .filter(idx -> arguments.get(idx).keywordArgument() != null)
      .findFirst();

    // Check the keyword arguments - build a set of all provided keyword arguments and check if all named fields have
    // a match in this set.
    Set<String> kwArguments = new HashSet<>();
    if (firstKwIdx.isPresent()) {
      arguments.subList(firstKwIdx.getAsInt(), arguments.size()).forEach(argument -> kwArguments.add(argument.keywordArgument().name()));
    }
    format.get().replacementFields().stream()
      .filter(field -> field.isNamed() && !kwArguments.contains(field.name()))
      .forEach(field -> reportIssue(ctx, qualifier, literal, String.format("Provide a value for field \"%s\".", field.name())));

    // Produce a list of unmatched positional indices and re-use it for the issue message.
    // We basically want to see if there is a position in the field list that is larger than the number of
    // the positional arguments provided.
    int firstIdx = firstKwIdx.orElse(arguments.size());
    String unmatchedPositionals = format.get().replacementFields().stream()
      .filter(field -> field.isPositional() && field.position() >= firstIdx)
      .map(field -> String.valueOf(field.position()))
      .distinct()
      .collect(Collectors.joining(", "));

    if (!unmatchedPositionals.isEmpty()) {
      reportIssue(ctx, qualifier, literal, String.format("Provide a value for field(s) with index %s.", unmatchedPositionals));
    }
  }

  private static void checkPrintfStyle(SubscriptionContext ctx) {
    BinaryExpression expression = (BinaryExpression) ctx.syntaxNode();
    StringLiteral formatString = extractStringLiteral(expression.leftOperand());
    if (formatString == null) {
      return;
    }

    if (formatString.stringElements().stream().anyMatch(s -> s.prefix().toLowerCase(Locale.ENGLISH).contains("b"))) {
      // Do not bother with byte formatting for now.
      return;
    }

    Optional<StringFormat> formatOptional = StringFormat.createFromPrintfStyle(ctx, expression.leftOperand(), formatString);
    if (!formatOptional.isPresent()) {
      // The string format contains invalid syntax.
      return;
    }

    StringFormat format = formatOptional.get();
    Expression rhs = expression.rightOperand();
    if (format.numExpectedArguments() == 0) {
      // The format does not contain any replacement fields, but with a mapping or a list as RHS, it won't result in a runtime error.
      if (!isMapping(rhs) && !rhs.type().canOnlyBe(BuiltinTypes.LIST)) {
        reportIssue(ctx, expression.leftOperand(), formatString, "Add replacement field(s) to this formatted string.");
      }
      return;
    }

    if (format.hasNamedFields()) {
      checkNamed(ctx, format, rhs);
    } else {
      checkPositional(ctx, format, rhs);
    }
  }

  private static void checkNamed(SubscriptionContext ctx, StringFormat format, Expression rhs) {
    if (rhs.is(Tree.Kind.DICTIONARY_LITERAL)) {
      checkDictionaries(ctx, format, ((DictionaryLiteral) rhs));
    } else if (!isMapping(rhs)) {
      ctx.addIssue(rhs, "Replace this formatting argument with a mapping.");
    }
  }

  private static void checkPositional(SubscriptionContext ctx, StringFormat format, Expression rhs) {
    if (rhs.is(Tree.Kind.TUPLE)) {
      checkTuples(ctx, format, ((Tuple) rhs));
    } else if (format.numExpectedArguments() == 1) {
      format.replacementFields().get(0).validateArgument(rhs);
    } else if (!rhs.type().canBeOrExtend("tuple")) {
      // Positional fields require tuples
      ctx.addIssue(rhs, "Replace this formatting argument with a tuple.");
    }
  }

  private static void checkTuples(SubscriptionContext ctx, StringFormat format, Tuple tuple) {
    if (tuple.elements().stream().anyMatch(expression -> expression.is(Tree.Kind.UNPACKING_EXPR))) {
      return;
    }

    if (format.numExpectedArguments() != tuple.elements().size()) {
      reportInvalidArgumentSize(ctx, tuple, format.numExpectedArguments(), tuple.elements().size());
      return;
    }

    for (int i = 0; i < tuple.elements().size(); ++i) {
      format.replacementFields().get(i).validateArgument(tuple.elements().get(i));
    }
  }

  private static void checkDictionaries(SubscriptionContext ctx, StringFormat format, DictionaryLiteral dict) {
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

      fields.forEach(field -> field.validateArgument(pair.value()));
    }

    // Check if we have any unmatched field names left
    fieldMap.keySet().forEach(fieldName -> ctx.addIssue(dict, String.format("Provide a value for field \"%s\".", fieldName)));
  }

  private static boolean isMapping(Expression expression) {
    // We consider everything having __getitem__ a mapping, with the exception of list and tuple.
    return NOT_MAPPING_TYPES.stream().noneMatch(type -> expression.type().canOnlyBe(type))
      && expression.type().canHaveMember("__getitem__");
  }

  private static void reportInvalidArgumentSize(SubscriptionContext ctx, Tree tree, long expected, long actual) {
    if (expected > actual) {
      ctx.addIssue(tree, String.format("Add %d missing argument(s).", expected - actual));
    } else {
      ctx.addIssue(tree, String.format("Remove %d unexpected argument(s).", actual - expected));
    }
  }

  private static StringLiteral extractStringLiteral(Tree tree) {
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

  private static void reportIssue(SubscriptionContext ctx, Tree primary, Tree secondary, String message) {
    PythonCheck.PreciseIssue preciseIssue = ctx.addIssue(primary, message);
    if (primary != secondary) {
      preciseIssue.secondary(secondary, null);
    }
  }
}
