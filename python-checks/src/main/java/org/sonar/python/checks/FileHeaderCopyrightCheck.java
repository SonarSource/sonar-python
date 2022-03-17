/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2022 SonarSource SA
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
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.sonar.check.Rule;
import org.sonar.check.RuleProperty;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Trivia;

import static org.sonar.python.tree.TreeUtils.getTextFromComments;
import static org.sonar.python.tree.TreeUtils.groupTrivias;

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

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> {
      if (isRegularExpression) {
        if (searchPattern == null) {
          try {
            searchPattern = Pattern.compile(headerFormat, Pattern.DOTALL);
          } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("[" + getClass().getSimpleName() + "] Unable to compile the regular expression: " + headerFormat, e);
          }
        }
      }

      Token token = ctx.syntaxNode().firstToken();
      String retrievedText = getDocstringLines(ctx.syntaxNode());
      if(retrievedText.equals("")){
        List<List<Trivia>> groupedTrivias = groupTrivias(token);
        if(!groupedTrivias.isEmpty()){
          retrievedText = getTextFromComments(groupedTrivias.get(0));
        }
      }

      if (isRegularExpression) {
        checkRegularExpression(ctx, retrievedText);
      } else {
        if(!headerFormat.isEmpty() && !retrievedText.startsWith(headerFormat)){
          ctx.addFileIssue(MESSAGE);
        }
      }
    });
  }

  private void checkRegularExpression(SubscriptionContext ctx, String fileContent) {
    Matcher matcher = searchPattern.matcher(fileContent);
    if (!matcher.find() || matcher.start() != 0) {
      ctx.addFileIssue(MESSAGE);
    }
  }

  private static String getDocstringLines(Tree token){
    StringLiteral docstring = ((FileInput)token).docstring();
    if(docstring != null){
      return docstring.firstToken().value()
        .replace("\"\"\"", "")
        .replaceAll("\\n[^\\S\\r\\n]+", "\n");
      // Remove any white space after \n, but not another \n itself nor carriage-return
    }
    return "";
  }
}
