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

import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.OptionalInt;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.DictionaryLiteral;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tuple;
import org.sonar.plugins.python.api.types.BuiltinTypes;

@Rule(key = "S2275")
public class StringFormatMisuseCheck extends AbstractStringFormatCheck {

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.MODULO, this::checkPrintfStyle);
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, this::checkStrFormatStyle);
  }

  @Override
  protected void checkPrintfStyle(SubscriptionContext ctx, BinaryExpression modulo, StringLiteral literal) {
    Optional<StringFormat> formatOptional = StringFormat.createFromPrintfStyle(
      syntaxIssueReporter(ctx, modulo.leftOperand(), literal), literal.trimmedQuotesValue());
    if (!formatOptional.isPresent()) {
      // The string format contains invalid syntax.
      return;
    }

    StringFormat format = formatOptional.get();
    Expression rhs = modulo.rightOperand();
    if (format.numExpectedArguments() == 0) {
      // The format does not contain any replacement fields, but with a mapping or a list as RHS, it won't result in a runtime error.
      if (!isMapping(rhs) && !rhs.type().canOnlyBe(BuiltinTypes.LIST)) {
        reportIssue(ctx, modulo.leftOperand(), literal, "Add replacement field(s) to this formatted string.");
      }
      return;
    }

    if (format.hasNamedFields()) {
      checkNamed(ctx, format, rhs);
    } else {
      checkPositional(ctx, format, rhs);
    }
  }

  @Override
  protected void checkStrFormatStyle(SubscriptionContext ctx, CallExpression callExpression, Expression qualifier, StringLiteral literal) {
    // Check the arguments for out of scope cases before we try to parse the string
    if (callExpression.arguments().stream().anyMatch(argument -> !argument.is(Tree.Kind.REGULAR_ARGUMENT))) {
      return;
    }

    Optional<StringFormat> format = StringFormat.createFromStrFormatStyle(syntaxIssueReporter(ctx, qualifier, literal), literal.trimmedQuotesValue());
    if (!format.isPresent()) {
      return;
    }

    List<RegularArgument> arguments = callExpression.arguments().stream()
      .map(RegularArgument.class::cast)
      .toList();

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

  private static void checkNamed(SubscriptionContext ctx, StringFormat format, Expression rhs) {
    if (rhs.is(Tree.Kind.DICTIONARY_LITERAL)) {
      checkPrintfDictionary(ctx, format, ((DictionaryLiteral) rhs));
    } else if (!isMapping(rhs)) {
      ctx.addIssue(rhs, "Replace this formatting argument with a mapping.");
    }
  }

  private static void checkPositional(SubscriptionContext ctx, StringFormat format, Expression rhs) {
    if (rhs.is(Tree.Kind.TUPLE)) {
      checkTuples(ctx, format, ((Tuple) rhs));
    } else if (format.numExpectedArguments() == 1) {
      format.replacementFields().get(0).validateArgument(ctx, rhs);
    } else if (!rhs.type().canBeOrExtend("tuple")) {
      // Positional fields require tuples
      ctx.addIssue(rhs, "Replace this formatting argument with a tuple.");
    }
  }

  private static void checkTuples(SubscriptionContext ctx, StringFormat format, Tuple tuple) {
    if (tuple.elements().stream().anyMatch(expression -> expression.is(Tree.Kind.UNPACKING_EXPR))) {
      return;
    }

    checkPrintfExpressionList(ctx, format, tuple.firstToken(), tuple.lastToken(), tuple.elements());
  }
}
