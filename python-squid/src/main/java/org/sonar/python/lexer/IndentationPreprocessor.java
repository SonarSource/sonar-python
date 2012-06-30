/*
 * Sonar Python Plugin
 * Copyright (C) 2011 Waleri Enns
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
import java.util.Stack;

/**
 * http://docs.python.org/release/3.2/reference/lexical_analysis.html#indentation
 */
public class IndentationPreprocessor extends Preprocessor {

  private final Stack<Integer> stack;

  public IndentationPreprocessor(Stack<Integer> indentationStack) {
    this.stack = indentationStack;
  }

  @Override
  public void init() {
    stack.clear();
    stack.push(0);
  }

  @Override
  public PreprocessorAction process(List<Token> tokens) {
    Token token = tokens.get(0);
    if (token.getType() == GenericTokenType.EOF) {
      if (stack.isEmpty()) {
        return PreprocessorAction.NO_OPERATION;
      }

      List<Token> tokensToInject = Lists.newArrayList();
      while (stack.peek() > 0) {
        stack.pop();
        tokensToInject.add(Token.builder(token)
            .setType(PythonTokenType.DEDENT)
            .setValueAndOriginalValue("")
            .build());
      }
      return new PreprocessorAction(0, Collections.EMPTY_LIST, tokensToInject);
    }
    return PreprocessorAction.NO_OPERATION;
  }

}
