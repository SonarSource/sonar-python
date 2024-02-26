/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
      switch (twoCharactersBefore) {
        case '`':
        case '*':
        case '_':
        case '|':
          return true;
        default:
          return false;
      }
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
