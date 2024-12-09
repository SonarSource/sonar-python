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

import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.IssueLocation;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.plugins.python.api.PythonLine;
import org.sonar.plugins.python.api.PythonVisitorContext;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
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
        var pythonLineNumber = new PythonLine(i + 1);
        PreciseIssue issue = new PreciseIssue(this, IssueLocation.atLineLevel(MESSAGE, pythonLineNumber.line(), ctx.pythonFile()));

        issue.addQuickFix(PythonQuickFix.newQuickFix("Remove trailing whitespaces")
          .addTextEdit(TextEditUtils.removeRange(pythonLineNumber, matcher.start(), pythonLineNumber, matcher.end()))
          .build());

        ctx.addIssue(issue);
      }
    }
  }
}

