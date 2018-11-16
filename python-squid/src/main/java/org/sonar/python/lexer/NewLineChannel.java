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
package org.sonar.python.lexer;

import com.sonar.sslr.api.Token;
import com.sonar.sslr.impl.Lexer;
import org.sonar.python.api.PythonTokenType;
import org.sonar.sslr.channel.Channel;
import org.sonar.sslr.channel.CodeReader;

/**
 * http://docs.python.org/reference/lexical_analysis.html#explicit-line-joining
 * http://docs.python.org/reference/lexical_analysis.html#implicit-line-joining
 * http://docs.python.org/reference/lexical_analysis.html#blank-lines
 */
public class NewLineChannel extends Channel<Lexer> {

  private final LexerState lexerState;

  public NewLineChannel(LexerState lexerState) {
    this.lexerState = lexerState;
  }

  @Override
  public boolean consume(CodeReader code, Lexer output) {
    char ch = (char) code.peek();
    checkForBrackets(ch);

    if ((ch == '\\') && isNewLine(code.charAt(1))) {
      // Explicit line joining
      code.pop();
      joinLines(code);
      return true;
    }

    if (isNewLine(ch)) {
      processNewLine(code, output);
      return true;
    }

    return false;
  }

  private boolean processNewLine(CodeReader code, Lexer output) {
    if (isImplicitLineJoining()) {
      // Implicit line joining
      joinLines(code);
      return true;
    }

    if (output.getTokens().isEmpty() || (output.getTokens().get(output.getTokens().size() - 1).getType().equals(PythonTokenType.NEWLINE))) {
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
        .setGeneratedCode(true)
        .build());
    consumeEOL(code);
    return false;
  }

  private void checkForBrackets(char ch) {
    switch (ch) {
      case '[':
      case '(':
      case '{':
        lexerState.brackets++;
        break;
      case ']':
      case ')':
      case '}':
        lexerState.brackets--;
        break;
      default:
        break;
    }
  }

  private void joinLines(CodeReader code) {
    while (Character.isWhitespace(code.peek())) {
      code.pop();
    }
    lexerState.joined = true;
  }

  private static void consumeEOL(CodeReader code) {
    if ((code.charAt(0) == '\r') && (code.charAt(1) == '\n')) {
      // \r\n
      code.pop();
      code.pop();
    } else {
      // \r or \n
      code.pop();
    }
  }

  private static boolean isNewLine(char ch) {
    return (ch == '\n') || (ch == '\r');
  }

  private boolean isImplicitLineJoining() {
    return lexerState.brackets > 0;
  }

}
