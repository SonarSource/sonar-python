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

import java.util.ArrayList;
import java.util.List;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.check.RuleProperty;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Trivia;

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


  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> {
      // Can there be no firstToken ?
      Token token = ctx.syntaxNode().firstToken();
      List<List<Trivia>> groupedTrivias = groupTrivias(token);
      String retrievedString;
      if(!groupedTrivias.isEmpty()){
        retrievedString = getTextFromComments(groupedTrivias.get(0));
      }else{
        retrievedString= getDocstringLines(ctx.syntaxNode());
      }
      if(!headerFormat.isEmpty() && !retrievedString.startsWith(headerFormat)){
        ctx.addFileIssue(MESSAGE);
      }
    });
  }

  private String getDocstringLines(Tree token){
    StringLiteral docstring = ((FileInput)token).docstring();
    if(docstring != null){
      return docstring.firstToken().value()
        .replace("\"\"\"", "")
        .replaceAll("\\n[^\\S\\r\\n]+", "\n");
      // Remove any white space after \n, but not another \n itself nor carriage-return
    }
    return "";
  }

  private static List<List<Trivia>> groupTrivias(Token token) {
    List<List<Trivia>> result = new ArrayList<>();
    List<Trivia> currentGroup = null;
    for (Trivia trivia : token.trivia()) {
      currentGroup = handleOneLineComment(result, currentGroup, trivia);
    }
    if (currentGroup != null) {
      result.add(currentGroup);
    }
    return result;
  }

  private static String getTextFromComments(List<Trivia> triviaGroup) {
    StringBuilder commentTextSB = new StringBuilder();
    for (Trivia trivia : triviaGroup) {
      String value = trivia.value();
      while (value.startsWith("#") || value.startsWith(" #")) {
        value = value.substring(1);
      }
      if (value.startsWith(" ")) {
        value = value.substring(1);
      }
      if (triviaGroup.size() == 1) {
        value = value.trim();
      }
      if (!isOneWord(value)) {
        commentTextSB.append(value);
        commentTextSB.append("\n");
      }
    }
    return commentTextSB.toString();
  }

  private static boolean isOneWord(String text) {
    return text.matches("\\s*[\\w/\\-]+\\s*#*\n*");
  }


  private static List<Trivia> handleOneLineComment(List<List<Trivia>> result, @Nullable List<Trivia> currentGroup, Trivia trivia) {
    List<Trivia> newTriviaGroup = currentGroup;
    if (currentGroup == null) {
      newTriviaGroup = new ArrayList<>();
      newTriviaGroup.add(trivia);
    } else if (currentGroup.get(currentGroup.size() - 1).token().line() + 1 == trivia.token().line()) {
      newTriviaGroup.add(trivia);
    } else {
      result.add(currentGroup);
      newTriviaGroup = new ArrayList<>();
      newTriviaGroup.add(trivia);
    }
    return newTriviaGroup;
  }
}
