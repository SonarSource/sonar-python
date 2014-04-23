/*
 * SonarQube Python Plugin
 * Copyright (C) 2011 SonarSource and Waleri Enns
 * dev@sonar.codehaus.org
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
 * You should have received a copy of the GNU Lesser General Public
 * License along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02
 */
package org.sonar.python.lexer;

import com.google.common.collect.Lists;
import com.sonar.sslr.api.GenericTokenType;
import com.sonar.sslr.api.Preprocessor;
import com.sonar.sslr.api.PreprocessorAction;
import com.sonar.sslr.api.Token;
import org.sonar.python.api.PythonTokenType;

import java.util.Collections;
import java.util.List;

/**
 * http://docs.python.org/reference/lexical_analysis.html#indentation
 */
public class IndentationPreprocessor extends Preprocessor {

  private final LexerState lexerState;

  public IndentationPreprocessor(LexerState lexerState) {
    this.lexerState = lexerState;
  }

  @Override
  public void init() {
    lexerState.reset();
  }

  @Override
  public PreprocessorAction process(List<Token> tokens) {
    Token token = tokens.get(0);
    if (token.getType() == GenericTokenType.EOF) {
      if (lexerState.indentationStack.isEmpty()) {
        return PreprocessorAction.NO_OPERATION;
      }

      List<Token> tokensToInject = Lists.newArrayList();
      while (lexerState.indentationStack.peek() > 0) {
        lexerState.indentationStack.pop();
        tokensToInject.add(Token.builder(token)
            .setURI(token.getURI())
            .setType(PythonTokenType.DEDENT)
            .setLine(token.getLine())
            .setColumn(token.getColumn())
            .setValueAndOriginalValue("")
            .build());
      }
      return new PreprocessorAction(0, Collections.EMPTY_LIST, tokensToInject);
    }
    return PreprocessorAction.NO_OPERATION;
  }

}
