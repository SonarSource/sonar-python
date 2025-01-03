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

import java.util.Objects;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import javax.annotation.Nullable;
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
  private static final String ADD_HEADER_MESSAGE = "Add a header to this file.";
  private static final String UPDATE_HEADER_MESSAGE = "Update the header of this file to match the expected one.";

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

      if(headerFormat.isEmpty()) {
        return;
      }

      String header = getHeaderText(ctx);
      String fileContent = ctx.pythonFile().content();
      var fileContentWithoutShebang = shebangPattern.matcher(fileContent).replaceFirst("");

      if (header == null && !fileContentWithoutShebang.startsWith("#")) {
        ctx.addFileIssue(ADD_HEADER_MESSAGE);
      } else if (!isStartingWithCopyrightHeader(header, fileContentWithoutShebang)) {
        ctx.addFileIssue(UPDATE_HEADER_MESSAGE);
      }
    });
  }

  private static @Nullable String getHeaderText(SubscriptionContext ctx) {
    StringLiteral tokenDoc = ((FileInput) ctx.syntaxNode()).docstring();
    if (tokenDoc != null && tokenDoc.firstToken().line() == 1) {
      return tokenDoc.firstToken().value();
    }
    return null;
  }

  private boolean isStartingWithCopyrightHeader(@Nullable String header, String fileContentWithoutShebang) {
    if (isRegularExpression) {
      return isStartingWithRegexSearchPattern(header, fileContentWithoutShebang);
    } else {
      return isStartingWithNormalSearchPattern(header, fileContentWithoutShebang);
    }
  }

  private boolean isStartingWithRegexSearchPattern(String... fileContent) {
    return Stream.of(fileContent)
      .filter(Objects::nonNull)
      .map(searchPattern::matcher)
      .anyMatch(matcher -> matcher.find() && matcher.start() == 0);
  }

  private boolean isStartingWithNormalSearchPattern(String... fileContent) {
    return Stream.of(fileContent)
      .filter(Objects::nonNull)
      .anyMatch(content -> content.startsWith(headerFormat));
  }
}
