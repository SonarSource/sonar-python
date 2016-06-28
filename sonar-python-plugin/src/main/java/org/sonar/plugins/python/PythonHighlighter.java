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
 * Note that docstrings are handled (i.e., colored) like normal string literals.
 */
public class PythonHighlighter extends SquidAstVisitor<Grammar> implements AstAndTokenVisitor {

  private NewHighlighting newHighlighting;

  private final SensorContext context;

  public PythonHighlighter(SensorContext context) {
    this.context = context;
  }

  @Override
  public void visitFile(@Nullable AstNode astNode) {
    newHighlighting = context.newHighlighting();
    InputFile inputFile = context.fileSystem().inputFile(context.fileSystem().predicates().is(getContext().getFile().getAbsoluteFile()));
    newHighlighting.onFile(inputFile);
  }

  @Override
  public void visitToken(Token token) {
    if (token.getType().equals(PythonTokenType.STRING)) {
      // string literals, including doc string
      highlight(token, TypeOfText.STRING);

    } else if (token.getType().equals(PythonTokenType.NUMBER)) {
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

}
