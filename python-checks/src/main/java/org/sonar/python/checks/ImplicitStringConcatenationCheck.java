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

import java.util.List;
import java.util.regex.Pattern;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.StringElement;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;

import static org.sonar.python.quickfix.TextEditUtils.insertAfter;
import static org.sonar.python.quickfix.TextEditUtils.replaceRange;

@Rule(key = "S5799")
public class ImplicitStringConcatenationCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE_SINGLE_LINE = "Merge these implicitly concatenated strings; or did you forget a comma?";
  private static final String MESSAGE_MULTIPLE_LINES = "Add a \"+\" operator to make the string concatenation explicit; or did you forget a comma?";
  // Column beyond which we assume the concatenation to be done intentionally for readability
  private static final int MAX_COLUMN = 65;
  // Won't report on line ending or starting with either \n, spaces or any punctuation
  private static final Pattern END_LINE_PATTERN = Pattern.compile("^.*(\\\\n|\\s|\\p{IsPunct})$");
  private static final Pattern START_LINE_PATTERN = Pattern.compile("^(\\\\n|\\s|\\p{IsPunct}).*");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.STRING_LITERAL, ctx -> {
      StringLiteral stringLiteral = (StringLiteral) ctx.syntaxNode();
      if (stringLiteral.parent().is(Tree.Kind.MODULO, Tree.Kind.QUALIFIED_EXPR)) {
        // if string formatting is used, explicit string concatenation with "+" might fail
        return;
      }
      if (stringLiteral.stringElements().size() == 1) {
        return;
      }
      checkStringLiteral(stringLiteral, ctx);
    });
  }

  private static void checkStringLiteral(StringLiteral stringLiteral, SubscriptionContext ctx) {
    List<StringElement> stringElements = stringLiteral.stringElements();
    for (int i = 1; i < stringElements.size(); i++) {
      StringElement current = stringElements.get(i);
      StringElement previous = stringElements.get(i - 1);
      if (!current.prefix().equalsIgnoreCase(previous.prefix()) || !haveSameQuotes(current, previous)) {
        continue;
      }
      if (current.firstToken().line() == previous.firstToken().line()) {
        createQuickFix(ctx.addIssue(previous.firstToken(), MESSAGE_SINGLE_LINE).secondary(current.firstToken(), null), previous, current);
        // Only raise 1 issue per string literal
        return;
      }
      if ((isWithinCollection(stringLiteral) && !isException(previous, current))) {
        createQuickFix(ctx.addIssue(previous.firstToken(), MESSAGE_MULTIPLE_LINES).secondary(current.firstToken(), null), previous, current);
        return;
      }
    }
  }

  private static boolean isException(StringElement first, StringElement second) {
    if (first.firstToken().column() + first.value().length() > MAX_COLUMN) {
      return true;
    }
    return END_LINE_PATTERN.matcher(first.trimmedQuotesValue()).matches() || START_LINE_PATTERN.matcher(second.trimmedQuotesValue()).matches();
  }

  private static boolean isWithinCollection(StringLiteral stringLiteral) {
    return stringLiteral.parent().is(Tree.Kind.TUPLE, Tree.Kind.EXPRESSION_LIST);
  }

  private static boolean haveSameQuotes(StringElement first, StringElement second) {
    return first.isTripleQuoted() == second.isTripleQuoted() &&
      first.value().charAt(first.value().length() - 1) == second.value().charAt(second.value().length() - 1);
  }

  private static boolean isInFunctionOrArrayOrTupleOrExpressionOrSet(StringElement token) {
    Tree t = token;
    while (t.parent().is(Tree.Kind.STRING_LITERAL)) {
      t = t.parent();
    }
    Tree parent = t.parent();

    return parent.is(Tree.Kind.EXPRESSION_LIST, Tree.Kind.PLUS, Tree.Kind.REGULAR_ARGUMENT,
      Tree.Kind.SET_LITERAL, Tree.Kind.TUPLE);
  }

  private static void createQuickFix(PreciseIssue issue, StringElement start, StringElement end) {
    String textStart = start.value();
    String textEnd = end.value();

    if (isInFunctionOrArrayOrTupleOrExpressionOrSet(start)) {
      PythonQuickFix quickFix = PythonQuickFix.newQuickFix("Add the comma between string or byte tokens.")
        .addTextEdit(insertAfter(start, ","))
        .build();
      issue.addQuickFix(quickFix);
    }

    PythonQuickFix quickFix = PythonQuickFix.newQuickFix("Make the addition sign between string or byte tokens explicit.")
      .addTextEdit(replaceRange(start, end, textStart + " + " + textEnd))
      .build();
    issue.addQuickFix(quickFix);
  }
}
