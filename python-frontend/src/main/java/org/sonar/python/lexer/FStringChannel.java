/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2020 SonarSource SA
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
    if (lexerState.brackets == 0) {
      return consumeNonExpressionChars(code, output);
    } else if (lexerState.brackets == 1) {
      char c = code.charAt(0);
      if (c == '}') {
        code.pop();
        lexerState.brackets = 0;
        return true;
      } else if (c == '!') {
        code.pop();
        code.pop();
      }
    }
    return false;
  }

  private void setInitialLineAndColumn(CodeReader code) {
    if (code.getLinePosition() == 1 && code.getColumnPosition() == 0) {
      code.setLinePosition(lexerState.initialLine);
      code.setColumnPosition(lexerState.initialColumn);
    }
  }

  private boolean consumeNonExpressionChars(CodeReader code, Lexer output) {
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
    if (code.charAt(0) == '{') {
      sb.append((char) code.pop());
      lexerState.brackets = 1;
    }
    if (sb.length() != 0) {
      output.addToken(Token.builder()
        .setType(GenericTokenType.UNKNOWN_CHAR)
        .setValueAndOriginalValue(sb.toString())
        .setURI(output.getURI())
        .setLine(line)
        .setColumn(column)
        .build());
      sb.setLength(0);
      return true;
    }
    return false;
  }
}
