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

import java.util.regex.Pattern;
import java.util.stream.Stream;
import org.sonar.check.Rule;
import org.sonar.check.RuleProperty;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;

@Rule(key = "S1451")
public class FileHeaderCopyrightCheck extends PythonSubscriptionCheck {

  private static final String DEFAULT_HEADER_FORMAT = "";
  private static final String MESSAGE = "Add or update the header of this file.";

  @RuleProperty(
    key = "headerFormat",
    description = "Expected copyright and license header",
    defaultValue = DEFAULT_HEADER_FORMAT,
    type = "TEXT")
  public String headerFormat = DEFAULT_HEADER_FORMAT;

  @RuleProperty(
    key = "isRegularExpression",
    description = "Whether the headerFormat is a regular expression",
    defaultValue = "false")
  public boolean isRegularExpression = false;
  private Pattern searchPattern = null;
  private Pattern shebangPattern = Pattern.compile("^#![^\\n]+\\n", Pattern.MULTILINE);

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> {
      if (isRegularExpression && searchPattern == null) {
        try {
          searchPattern = Pattern.compile(headerFormat, Pattern.DOTALL);
        } catch (IllegalArgumentException e) {
          throw new IllegalArgumentException("[" + getClass().getSimpleName() + "] Unable to compile the regular expression: " + headerFormat, e);
        }
      }

      var header = getHeaderText(ctx);
      var headerWithoutShebang = shebangPattern.matcher(header).replaceFirst("");

      if (isRegularExpression) {
        checkRegularExpression(ctx, header, headerWithoutShebang);
      } else if (!headerFormat.isEmpty()) {
        var matches = Stream.of(header, headerWithoutShebang)
          .anyMatch(h -> h.startsWith(headerFormat));

        if (!matches) {
          ctx.addFileIssue(MESSAGE);
        }
      }
    });
  }

  private static String getHeaderText(SubscriptionContext ctx) {
    StringLiteral tokenDoc = ((FileInput) ctx.syntaxNode()).docstring();
    if (tokenDoc != null && tokenDoc.firstToken().line() == 1) {
      return tokenDoc.firstToken().value();
    }
    return ctx.pythonFile().content();
  }

  private void checkRegularExpression(SubscriptionContext ctx, String... fileContent) {
    var matches = Stream.of(fileContent)
      .map(searchPattern::matcher)
      .anyMatch(matcher -> matcher.find() && matcher.start() == 0);

    if (!matches) {
      ctx.addFileIssue(MESSAGE);
    }
  }

}
