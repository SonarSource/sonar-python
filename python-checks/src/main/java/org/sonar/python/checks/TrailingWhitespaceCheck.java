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

import java.util.Optional;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.IssueLocation;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.plugins.python.api.PythonVisitorContext;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.tree.TreeUtils;

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
        int r;
        if (!ctx.pythonFile().fileName().endsWith(".ipynb")) {
          r = lineNumber;
        } else {
          var rOpt = findPythonLineFromPhysicalLine(lineNumber, ctx.rootTree());
          if (rOpt.isEmpty()) {
            continue;
          }
          r = findPythonLineFromPhysicalLine(lineNumber, ctx.rootTree()).get().firstToken().line();
        }
        PreciseIssue issue = new PreciseIssue(this, IssueLocation.atLineLevel(MESSAGE, r));

        issue.addQuickFix(PythonQuickFix.newQuickFix("Remove trailing whitespaces")
          .addTextEdit(TextEditUtils.removeRange(r, matcher.start(), r, matcher.end()))
          .build());

        ctx.addIssue(issue);
      }
    }
  }

  private Optional<Tree> findPythonLineFromPhysicalLine(int physicalLine, FileInput fileInput) {
    return TreeUtils.firstChild(fileInput, t -> t.firstToken().pythonLine() == physicalLine);
  }
}

