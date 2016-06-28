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
import com.sonar.sslr.api.GenericTokenType;
import com.sonar.sslr.api.Grammar;
import com.sonar.sslr.api.Token;
import com.sonar.sslr.api.Trivia;
import javax.annotation.Nullable;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.api.batch.sensor.highlighting.NewHighlighting;
import org.sonar.api.batch.sensor.highlighting.TypeOfText;
import org.sonar.python.api.PythonKeyword;
import org.sonar.python.api.PythonTokenType;
import org.sonar.squidbridge.SquidAstVisitor;

/**
 * Colorizes Python code. Currently colorizes:
 * <ul>
 *   <li>
 *     String literals. Example:
 *     <pre>
 *       "hello"
 *     </pre>
 *   </li>
 *   <li>
 *     Keywords. Example:
 *     <pre>
 *       def
 *     </pre>
 *   </li>
 *   <li>
 *     Comments. Example:
 *     <pre>
 *        # some comment
 *     </pre>
 *   </li>
 *   <li>
 *     Doc strings. Examples:
 *     <pre>
 *        """ a doc string"""
 *        ''' another doc string '''
 *     </pre>
 *   </li>
 * </ul>
 * Note that doc strings, such as:
 * <pre>
 *   """ a doc string"""
 *   ''' another doc string '''
 * </pre>
 * are not supported (i.e., not colorized) yet.
 */
public class PythonHighlighter extends SquidAstVisitor<Grammar> implements AstAndTokenVisitor {

  private NewHighlighting newHighlighting;
  
  private final SensorContext context;
  
  /**
   * Creates a highlighter for the specified context.
   */
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
      // case: string literal
      highlight(token, TypeOfText.STRING);
    } else if (token.getType() instanceof PythonKeyword) {
      // case: keyword
      highlight(token, TypeOfText.KEYWORD);
    } else if (token.getType().equals(GenericTokenType.COMMENT)) {
      // case: doc string
      highlight(token, TypeOfText.COMMENT);
    } 
    
    for (Trivia trivia : token.getTrivia()) {
      // case: comment
      highlight(trivia.getToken(), TypeOfText.COMMENT);
    }
  }

  @Override
  public void leaveFile(@Nullable AstNode astNode) {
    newHighlighting.save();
  }
  
  private void highlight(Token token, TypeOfText typeOfText) {
    newHighlighting.highlight(token.getLine(), token.getColumn(), token.getLine(), token.getColumn() + token.getValue().length(), typeOfText);
  }

}
