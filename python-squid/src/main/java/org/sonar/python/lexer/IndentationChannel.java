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

import com.sonar.sslr.api.Token;
import com.sonar.sslr.impl.Lexer;
import org.sonar.sslr.channel.Channel;
import org.sonar.sslr.channel.CodeReader;
import org.sonar.python.api.PythonTokenType;

/**
 * http://docs.python.org/reference/lexical_analysis.html#indentation
 */
public class IndentationChannel extends Channel<Lexer> {

  private final StringBuilder buffer = new StringBuilder();
  private final LexerState lexerState;

  public IndentationChannel(LexerState lexerState) {
    this.lexerState = lexerState;
  }

  @Override
  public boolean consume(CodeReader code, Lexer lexer) {
    if (lexerState.joined) {
      lexerState.joined = false;
      return false;
    }

    if (code.getColumnPosition() != 0) {
      return false;
    }

    int line = code.getLinePosition();
    int column = code.getColumnPosition();

    int index = 0;
    char ch = code.charAt(index);
    while ((ch == ' ') || (ch == '\t')) {
      index++;
      ch = code.charAt(index);
    }

    if ((ch == '\n') || (ch == '\r') || (ch == '#') || (ch == (char) -1)) {
      // Blank line
      return false;
    }

    buffer.setLength(0);
    int indentationLevel = 0;
    for (int i = 0; i < index; i++) {
      buffer.append((char) code.pop());
      indentationLevel++;
    }

    if (indentationLevel > lexerState.indentationStack.peek()) {
      lexerState.indentationStack.push(indentationLevel);
      lexer.addToken(Token.builder()
          .setType(PythonTokenType.INDENT)
          .setValueAndOriginalValue(buffer.toString())
          .setURI(lexer.getURI())
          .setLine(line)
          .setColumn(column)
          .build());
    } else if (indentationLevel < lexerState.indentationStack.peek()) {
      while (indentationLevel < lexerState.indentationStack.peek()) {
        lexerState.indentationStack.pop();
        lexer.addToken(Token.builder()
            .setType(PythonTokenType.DEDENT)
            .setValueAndOriginalValue(buffer.toString())
            .setURI(lexer.getURI())
            .setLine(line)
            .setColumn(column)
            .build());
      }
    }

    return buffer.length() != 0;
  }

}
