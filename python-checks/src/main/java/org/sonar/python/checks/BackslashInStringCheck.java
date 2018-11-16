/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2018 SonarSource SA
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

import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.AstNodeType;
import java.util.Collections;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.python.PythonCheck;
import org.sonar.python.api.PythonTokenType;

@Rule(key = "S1717")
public class BackslashInStringCheck extends PythonCheck {

  private static final String MESSAGE = "Remove this \"\\\", add another \"\\\" to escape it, or make this a raw string.";
  private static final String VALID_ESCAPED_CHARACTERS = "abfnrtvxnNrtuU\\'\"0123456789\n\r";

  @Override
  public Set<AstNodeType> subscribedKinds() {
    return Collections.singleton(PythonTokenType.STRING);
  }

  @Override
  public void visitNode(AstNode node) {
    String string = node.getTokenOriginalValue();
    int length = string.length();
    boolean isEscaped = false;
    boolean inPrefix = true;
    boolean isThreeQuotes = length > 5 && "\"\"".equals(string.substring(1, 3));
    for (int i = 0; i < length; i++) {
      char c = string.charAt(i);
      inPrefix = isInPrefix(inPrefix, c);
      if (inPrefix) {
        if (c == 'r' || c == 'R') {
          return;
        }
      } else {
        if (isEscaped && VALID_ESCAPED_CHARACTERS.indexOf(c) == -1 && !isBackslashedSpaceAfterInlineMarkup(isThreeQuotes, string, i, c)) {
          addIssue(node, MESSAGE);
        }
        isEscaped = c == '\\' && !isEscaped;
      }
    }
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

}

