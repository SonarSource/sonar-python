/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2023 SonarSource SA
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

import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.IssueLocation;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.plugins.python.api.PythonVisitorContext;
import org.sonar.python.quickfix.PythonQuickFix;
import org.sonar.python.quickfix.TextEditUtils;

@Rule(key = "S1131")
public class TrailingWhitespaceCheck implements PythonCheck {

  private static final String MESSAGE = "Remove the useless trailing whitespaces at the end of this line.";
  private static final Pattern TRAILING_WS = Pattern.compile("\\s\\s*+$");

  @Override
  public void scanFile(PythonVisitorContext ctx) {
    String[] lines = ctx.pythonFile().content().split("\r\n|\n|\r", -1);
    for (int i = 0; i < lines.length; i++) {
      Matcher matcher = TRAILING_WS.matcher(lines[i]);
      if (matcher.find()) {
        int lineNumber = i + 1;
        PreciseIssue issue = new PreciseIssue(this, IssueLocation.atLineLevel(MESSAGE, lineNumber));

        issue.addQuickFix(PythonQuickFix.newQuickFix("Remove trailing whitespaces")
            .addTextEdit(TextEditUtils.removeRange(lineNumber, matcher.start(), lineNumber, matcher.end()))
          .build());

        ctx.addIssue(issue);
      }
    }
  }
}

