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

import com.sonar.sslr.api.Token;
import com.sonar.sslr.impl.Lexer;
import org.sonar.channel.Channel;
import org.sonar.channel.CodeReader;
import org.sonar.python.api.PythonTokenType;

/**
 * http://docs.python.org/release/3.2/reference/lexical_analysis.html#explicit-line-joining
 * http://docs.python.org/release/3.2/reference/lexical_analysis.html#implicit-line-joining
 * http://docs.python.org/release/3.2/reference/lexical_analysis.html#blank-lines
 */
public class NewLineChannel extends Channel<Lexer> {

  private final LexerState lexerState;

  public NewLineChannel(LexerState lexerState) {
    this.lexerState = lexerState;
  }

  @Override
  public boolean consume(CodeReader code, Lexer output) {
    char ch = (char) code.peek();
    switch (ch) {
      case '[':
        lexerState.squareBrackets++;
        break;
      case ']':
        lexerState.squareBrackets--;
        break;
      case '(':
        lexerState.parentheses++;
        break;
      case ')':
        lexerState.parentheses--;
        break;
      case '{':
        lexerState.curlyBraces++;
        break;
      case '}':
        lexerState.curlyBraces--;
        break;
    }

    if (ch == '\\' && isNewLine(code.charAt(1))) {
      // Explicit line joining
      code.pop();
      while (Character.isWhitespace(code.peek())) {
        code.pop();
      }
      return true;
    }

    if (isNewLine(ch)) {
      if (isImplicitLineJoining()) {
        // Implicit line joining
        while (Character.isWhitespace(code.peek())) {
          code.pop();
        }
        return true;
      }

      if ((output.getTokens().size() == 0) || (output.getTokens().get(output.getTokens().size() - 1).getType() == PythonTokenType.NEWLINE)) {
        // Blank line
        consumeEOL(code);
        return true;
      }

      // NEWLINE token
      output.addToken(Token.builder()
          .setLine(code.getLinePosition())
          .setColumn(code.getColumnPosition())
          .setURI(output.getURI())
          .setType(PythonTokenType.NEWLINE)
          .setValueAndOriginalValue("\n")
          .build());
      consumeEOL(code);
      return true;
    }

    return false;
  }

  private static void consumeEOL(CodeReader code) {
    if (code.charAt(0) == '\r' && code.charAt(1) == '\n') {
      // \r\n
      code.pop();
      code.pop();
    } else {
      // \r or \n
      code.pop();
    }
  }

  private static boolean isNewLine(char ch) {
    return ch == '\n' || ch == '\r';
  }

  private boolean isImplicitLineJoining() {
    return lexerState.parentheses > 0 || lexerState.squareBrackets > 0 || lexerState.curlyBraces > 0;
  }

}
