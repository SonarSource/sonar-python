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

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.StringElement;
import org.sonar.plugins.python.api.tree.Tree;

@Rule(key = "S1717")
public class BackslashInStringCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Remove this \"\\\", add another \"\\\" to escape it, or make this a raw string.";
  private static final String VALID_ESCAPED_CHARACTERS = "abfnrtvxnNrtuU\\'\"0123456789\n\r";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.STRING_ELEMENT, ctx -> {
      StringElement pyStringLiteralTree = ((StringElement) ctx.syntaxNode());
      String string = pyStringLiteralTree.value();
      int length = string.length();
      boolean isEscaped = false;
      boolean inPrefix = true;
      boolean isThreeQuotes = length > 5 && "\"\"".equals(string.substring(1, 3));
      for (int i = 0; i < length; i++) {
        char c = string.charAt(i);
        inPrefix = isInPrefix(inPrefix, c);
        if (isRawStringLiteral(inPrefix, c)) {
          return;
        } else {
          if (isEscaped && VALID_ESCAPED_CHARACTERS.indexOf(c) == -1 && !isBackslashedSpaceAfterInlineMarkup(isThreeQuotes, string, i, c)) {
            ctx.addIssue(pyStringLiteralTree, MESSAGE);
          }
          isEscaped = c == '\\' && !isEscaped;
        }
      }
    });
  }

  private static boolean isBackslashedSpaceAfterInlineMarkup(boolean isThreeQuotes, String string, int position, char current) {
    if (isThreeQuotes && current == ' ' && position > 6) {
      char twoCharactersBefore = string.charAt(position - 2);
      return switch (twoCharactersBefore) {
        case '`', '*', '_', '|' -> true;
        default -> false;
      };
    }
    return false;
  }

  private static boolean isInPrefix(boolean wasInPrefix, char currentChar) {
    return wasInPrefix && currentChar != '"' && currentChar != '\'';
  }

  private static boolean isRawStringLiteral(boolean inPrefix, char c) {
    if (inPrefix) {
      return c == 'r' || c == 'R';
    }
    return false;
  }
}
