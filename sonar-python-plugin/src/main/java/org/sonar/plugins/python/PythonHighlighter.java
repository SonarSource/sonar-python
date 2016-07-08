/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2016 SonarSource SA
 * mailto:contact AT sonarsource DOT com
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
package org.sonar.plugins.python;

import com.sonar.sslr.api.AstAndTokenVisitor;
import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.Grammar;
import com.sonar.sslr.api.Token;
import com.sonar.sslr.api.Trivia;
import javax.annotation.Nullable;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.api.batch.sensor.highlighting.NewHighlighting;
import org.sonar.api.batch.sensor.highlighting.TypeOfText;
import org.sonar.python.TokenLocation;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.api.PythonKeyword;
import org.sonar.python.api.PythonTokenType;
import org.sonar.squidbridge.SquidAstVisitor;

/**
 * Colors Python code. Currently colors:
 * <ul>
 *   <li>
 *     String literals. Examples:
 *     <pre>
 *       "hello"
 *
 *       'hello'
 *
 *       """ hello
 *           hello again
 *       """
 *     </pre>
 *   </li>
 *   <li>
 *     Keywords. Example:
 *     <pre>
 *       def
 *     </pre>
 *   </li>
 *   <li>
 *     Numbers. Example:
 *     <pre>
 *        123
 *        123L
 *        123.45
 *        123.45e-10
 *        123+88.99J
 *     </pre>
 *     For a negative number, the "minus" sign is not colored.
 *   </li>
 *   <li>
 *     Comments. Example:
 *     <pre>
 *        # some comment
 *     </pre>
 *   </li>
 * </ul>
 * Docstrings are handled (i.e., colored) as structured comments, not as normal string literals.
 * "Attribute docstrings" and "additional docstrings" (see PEP 258) are handled as normal string literals.
 * Reminder: a docstring is a string literal that occurs as the first statement in a module,
 * function, class, or method definition.
 */
public class PythonHighlighter extends SquidAstVisitor<Grammar> implements AstAndTokenVisitor {

  private NewHighlighting newHighlighting;

  private final SensorContext context;

  public PythonHighlighter(SensorContext context) {
    this.context = context;
  }

  @Override
  public void init() {
    subscribeTo(
      PythonTokenType.STRING
    );
  }

  @Override
  public void visitFile(@Nullable AstNode astNode) {
    newHighlighting = context.newHighlighting();
    InputFile inputFile = context.fileSystem().inputFile(context.fileSystem().predicates().is(getContext().getFile().getAbsoluteFile()));
    newHighlighting.onFile(inputFile);
  }

  @Override
  public void visitNode(AstNode astNode) {
    Token token = astNode.getToken();
    if (isDocString(astNode)) {
      highlight(token, TypeOfText.STRUCTURED_COMMENT);
    } else {
      highlight(token, TypeOfText.STRING);
    }
  }

  @Override
  public void visitToken(Token token) {
    if (token.getType().equals(PythonTokenType.NUMBER)) {
      highlight(token, TypeOfText.CONSTANT);
    } else if (token.getType() instanceof PythonKeyword) {
      highlight(token, TypeOfText.KEYWORD);
    }

    for (Trivia trivia : token.getTrivia()) {
      highlight(trivia.getToken(), TypeOfText.COMMENT);
    }
  }

  @Override
  public void leaveFile(@Nullable AstNode astNode) {
    newHighlighting.save();
  }

  private void highlight(Token token, TypeOfText typeOfText) {
    TokenLocation tokenLocation = new TokenLocation(token);
    newHighlighting.highlight(tokenLocation.startLine(), tokenLocation.startLineOffset(), tokenLocation.endLine(), tokenLocation.endLineOffset(), typeOfText);
  }

  private static boolean isDocString(AstNode stringNode) {
    AstNode statement = stringNode.getFirstAncestor(PythonGrammar.STATEMENT);
    if (statement != null &&
      isInModuleOrFunctionOrClassOrMethod(statement) &&
      statement.equals(statement.getParent().getFirstChild(PythonGrammar.STATEMENT))) {
      return true;
    }
    return false;
  }

  private static boolean isInModuleOrFunctionOrClassOrMethod(AstNode statement) {
    AstNode parent = statement.getParent();
    if (parent.is(PythonGrammar.FILE_INPUT)) {
      return true;
    } else {
      return parent.getParent().is(PythonGrammar.FUNCDEF, PythonGrammar.CLASSDEF);
    }
  }

}
