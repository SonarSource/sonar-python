/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
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

import com.sonar.sslr.api.GenericTokenType;
import com.sonar.sslr.api.Token;
import com.sonar.sslr.impl.Lexer;
import org.sonar.sslr.channel.Channel;
import org.sonar.sslr.channel.CodeReader;

/**
 * A channel to handle the literal_char parts inside f-strings.
 * See https://docs.python.org/3/reference/lexical_analysis.html#f-strings
 */
public class FStringChannel extends Channel<Lexer> {

  private static final char EOF = (char) -1;

  private final LexerState lexerState;
  private final StringBuilder sb = new StringBuilder();

  public FStringChannel(LexerState lexerState) {
    this.lexerState = lexerState;
  }

  @Override
  public boolean consume(CodeReader code, Lexer output) {
    setInitialLineAndColumn(code);
    if (code.charAt(0) == '#') {
      // disable comments
      addUnknownCharToken("#", output, code.getLinePosition(), code.getColumnPosition());
      code.pop();
      return true;
    }
    if (lexerState.brackets == 0) {
      int line = code.getLinePosition();
      int column = code.getColumnPosition();
      while (code.charAt(0) != EOF) {
        char c = code.charAt(0);
        if (c != '{') {
          sb.append((char) code.pop());
        } else if (code.charAt(1) == '{') {
          sb.append((char) code.pop());
          sb.append((char) code.pop());
        } else {
          break;
        }
      }
      if (sb.length() != 0) {
        addUnknownCharToken(sb.toString(), output, line, column);
        sb.setLength(0);
        return true;
      }
    }
    return false;
  }

  private static void addUnknownCharToken(String value, Lexer output, int line, int column) {
    output.addToken(Token.builder()
      .setType(GenericTokenType.UNKNOWN_CHAR)
      .setValueAndOriginalValue(value)
      .setURI(output.getURI())
      .setLine(line)
      .setColumn(column)
      .build());
  }

  private void setInitialLineAndColumn(CodeReader code) {
    if (code.getLinePosition() == 1 && code.getColumnPosition() == 0) {
      code.setLinePosition(lexerState.initialLine);
      code.setColumnPosition(lexerState.initialColumn);
    }
  }
}
