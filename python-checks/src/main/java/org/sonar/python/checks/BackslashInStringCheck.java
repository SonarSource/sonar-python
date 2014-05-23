/*
 * SonarQube Python Plugin
 * Copyright (C) 2011 SonarSource and Waleri Enns
 * dev@sonar.codehaus.org
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
 * You should have received a copy of the GNU Lesser General Public
 * License along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02
 */
package org.sonar.python.checks;

import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.Grammar;
import org.sonar.check.BelongsToProfile;
import org.sonar.check.Priority;
import org.sonar.check.Rule;
import org.sonar.python.api.PythonTokenType;
import org.sonar.squidbridge.checks.SquidCheck;

@Rule(
  key = "S1717",
  priority = Priority.MAJOR)
@BelongsToProfile(title = CheckList.SONAR_WAY_PROFILE, priority = Priority.MAJOR)
public class BackslashInStringCheck extends SquidCheck<Grammar> {

  private static final String MESSAGE = "Remove this \"\\\", add another \"\\\" to escape it, or make this a raw string.";
  private static final String VALID_ESCAPED_CHARACTERS = "abfnrtvxnNrtuU\\'\"0123456789\n\r";

  @Override
  public void init() {
    subscribeTo(PythonTokenType.STRING);
  }

  @Override
  public void visitNode(AstNode node) {
    String string = node.getTokenOriginalValue();
    int length = string.length();
    boolean isEscaped = false;
    boolean inPrefix = true;
    for (int i = 0; i < length; i++) {
      char c = string.charAt(i);
      inPrefix = isInPrefix(inPrefix, c);
      if (inPrefix && (c == 'r' || c == 'R')) {
        return;
      }
      if (!inPrefix) {
        if (isEscaped && VALID_ESCAPED_CHARACTERS.indexOf(c) == -1) {
          getContext().createLineViolation(this, MESSAGE, node);
        }
        isEscaped = c == '\\' && !isEscaped;
      }
    }
  }

  private boolean isInPrefix(boolean wasInPrefix, char currentChar) {
    return wasInPrefix && currentChar != '"' && currentChar != '\'';
  }

}

