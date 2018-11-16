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
import com.sonar.sslr.api.Grammar;
import com.sonar.sslr.api.Token;
import com.sonar.sslr.api.Trivia;
import com.sonar.sslr.impl.Parser;
import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.python.PythonCheck;
import org.sonar.python.PythonConfiguration;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.api.PythonTokenType;
import org.sonar.python.parser.PythonParser;

@Rule(key = CommentedCodeCheck.CHECK_KEY)
public class CommentedCodeCheck extends PythonCheck {

  public static final String CHECK_KEY = "S125";
  public static final String MESSAGE = "Remove this commented out code.";
  private static final Parser<Grammar> parser = PythonParser.create(new PythonConfiguration(StandardCharsets.UTF_8));

  @Override
  public Set<AstNodeType> subscribedKinds() {
    return Collections.singleton(PythonTokenType.STRING);
  }

  @Override
  public void visitNode(AstNode astNode) {
    if (isMultilineComment(astNode)) {
      visitMultilineComment(astNode.getToken());
    }
  }

  @Override
  public void visitToken(Token token) {
    List<List<Trivia>> groupedTrivias = groupTrivias(token);
    for (List<Trivia> triviaGroup : groupedTrivias) {
      checkTriviaGroup(triviaGroup);
    }
  }

  private void visitMultilineComment(Token token) {
    String value = token.getValue();
    int startStringContent;
    if (value.endsWith("'''")) {
      startStringContent = value.indexOf("'''") + 3;
    } else {
      startStringContent = value.indexOf("\"\"\"") + 3;
    }
    int endStringContent = value.length() - 3;
    String text = value.substring(startStringContent, endStringContent);
    text = text.trim();
    if (!isEmpty(text) && isTextParsedAsCode(text)) {
      addIssue(token, MESSAGE);
    }

  }

  private static boolean isMultilineComment(AstNode node) {
    String str = node.getTokenValue();
    AstNode expressionStatement = node.getFirstAncestor(PythonGrammar.EXPRESSION_STMT);
    return (str.endsWith("'''") || str.endsWith("\"\"\"")) && expressionStatement != null && expressionStatement.getNumberOfChildren() == 1;
  }

  private void checkTriviaGroup(List<Trivia> triviaGroup) {
    String text = getTextForParsing(triviaGroup);
    if (isEmpty(text)) {
      return;
    }
    if (isTextParsedAsCode(text)) {
      addIssue(triviaGroup.get(0).getToken(), MESSAGE);
    }
  }

  private static String getTextForParsing(List<Trivia> triviaGroup) {
    StringBuilder commentTextSB = new StringBuilder();
    for (Trivia trivia : triviaGroup) {
      String value = trivia.getToken().getValue();
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

  private static boolean isEmpty(String text) {
    return text.matches("\\s*");
  }

  private static boolean isTextParsedAsCode(String text) {
    try {
      AstNode astNode = parser.parse(text);
      List<AstNode> expressions = astNode.getDescendants(PythonGrammar.EXPRESSION_STMT);
      return astNode.getNumberOfChildren() > 1 && !isSimpleExpression(expressions);
    } catch (Exception e) {
      return false;
    }
  }

  private static boolean isSimpleExpression(List<AstNode> expressionStatements) {
    if (expressionStatements.size() != 1) {
      return false;
    }
    AstNode expressionStatement = expressionStatements.get(0);
    return (expressionStatement.getNumberOfChildren() == 1 && expressionStatement.getFirstChild().is(PythonGrammar.TESTLIST_STAR_EXPR))
      || expressionStatement.hasDirectChildren(PythonGrammar.ANNASSIGN);
  }

  private static List<List<Trivia>> groupTrivias(Token token) {
    List<List<Trivia>> result = new LinkedList<>();
    List<Trivia> currentGroup = null;
    for (Trivia trivia : token.getTrivia()) {
      currentGroup = handleOneLineComment(result, currentGroup, trivia);
    }
    if (currentGroup != null) {
      result.add(currentGroup);
    }
    return result;
  }

  private static List<Trivia> handleOneLineComment(List<List<Trivia>> result, @Nullable List<Trivia> currentGroup, Trivia trivia) {
    List<Trivia> newTriviaGroup = currentGroup;
    if (currentGroup == null) {
      newTriviaGroup = new LinkedList<>();
      newTriviaGroup.add(trivia);
    } else if (currentGroup.get(currentGroup.size() - 1).getToken().getLine() + 1 == trivia.getToken().getLine()) {
      newTriviaGroup.add(trivia);
    } else {
      result.add(currentGroup);
      newTriviaGroup = new LinkedList<>();
      newTriviaGroup.add(trivia);
    }
    return newTriviaGroup;
  }
}
