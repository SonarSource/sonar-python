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
 * http://docs.python.org/reference/lexical_analysis.html#string-literals
 */
public class LongStringLiteralsChannel extends Channel<Lexer> {

  private final StringBuilder sb = new StringBuilder();

  private int index;
  private char ch;

  @Override
  public boolean consume(CodeReader code, Lexer output) {
    readStringPrefix(code);
    if (ch != '\'' && ch != '\"') {
      return false;
    }
    if (!isLookingOn(code, ch, index)) {
      return false;
    }
    index++;
    while (!isLookingOn(code, ch, index)) {
      if (code.charAt(index) == '\\') {
        // escape
        index++;
      }
      index++;
    }
    for (int i = 0; i < index + 3; i++) {
      sb.append((char) code.pop());
    }
    output.addToken(Token.builder()
        .setLine(code.getLinePosition())
        .setColumn(code.getColumnPosition())
        .setURI(output.getURI())
        .setValueAndOriginalValue(sb.toString())
        .setType(PythonTokenType.STRING)
        .build());
    sb.setLength(0);
    return true;
  }

  private void readStringPrefix(CodeReader code) {
    index = 0;
    ch = Character.toUpperCase(code.charAt(index));
    if (ch == 'U' || ch == 'B') {
      index++;
      ch = Character.toUpperCase(code.charAt(index));
    }
    if (ch == 'R') {
      index++;
      ch = code.charAt(index);
    }
  }

  private static boolean isLookingOn(CodeReader code, char ch, int index) {
    return (code.charAt(index) == ch) && (code.charAt(index + 1) == ch) && (code.charAt(index + 2) == ch);
  }

}
