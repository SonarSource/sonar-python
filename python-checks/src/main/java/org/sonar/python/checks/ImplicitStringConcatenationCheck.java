/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
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

import java.util.List;
import java.util.regex.Pattern;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.StringElement;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;

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
      StringElement previous = stringElements.get(i-1);
      if (!current.prefix().equalsIgnoreCase(previous.prefix()) || !haveSameQuotes(current, previous)) {
        continue;
      }
      if (current.firstToken().line() == previous.firstToken().line()) {
        ctx.addIssue(previous.firstToken(), MESSAGE_SINGLE_LINE).secondary(current.firstToken(), null);
        // Only raise 1 issue per string literal
        return;
      }
      if ((isWithinCollection(stringLiteral) && !isException(previous, current))) {
        ctx.addIssue(previous.firstToken(), MESSAGE_MULTIPLE_LINES).secondary(current.firstToken(), null);
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
}

