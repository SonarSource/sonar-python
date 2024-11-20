/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource SA.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.python.lexer;

import java.util.Set;

import org.sonar.python.api.PythonTokenType;
import org.sonar.sslr.channel.Channel;
import org.sonar.sslr.channel.CodeReader;

import com.sonar.sslr.api.Token;
import com.sonar.sslr.impl.Lexer;

/**
 * http://docs.python.org/reference/lexical_analysis.html#string-literals
 */
public class StringLiteralsChannel extends Channel<Lexer> {

  private static final char EOF = (char) -1;
  private static final Set<Character> PREFIX_CHARS = Set.of('R', 'U', 'B');

  private final StringBuilder sb = new StringBuilder();

  private int index;
  private char ch;

  @Override
  public boolean consume(CodeReader code, Lexer output) {
    int line = code.getLinePosition();
    int column = code.getColumnPosition();
    index = 0;
    readStringPrefix(code);
    if ((ch != '\'') && (ch != '\"')) {
      return false;
    }
    if (!read(code)) {
      return false;
    }
    for (int i = 0; i < index; i++) {
      sb.append((char) code.pop());
    }
    output.addToken(Token.builder()
      .setLine(line)
      .setColumn(column)
      .setURI(output.getURI())
      .setValueAndOriginalValue(sb.toString())
      .setType(PythonTokenType.STRING)
      .build());
    sb.setLength(0);
    return true;
  }

  private boolean read(CodeReader code) {
    if (isLookingOnLongString(code, ch, index)) {
      return readLongString(code);
    } else {
      return readString(code);
    }
  }

  private boolean readString(CodeReader code) {
    index++;
    while (code.charAt(index) != ch) {
      if (code.charAt(index) == EOF) {
        return false;
      }
      if (code.charAt(index) == '\\') {
        // escape
        index++;
      }
      index++;
    }
    index++;
    return true;
  }

  private boolean readLongString(CodeReader code) {
    index += 3;
    while (!isLookingOnLongString(code, ch, index)) {
      if (code.charAt(index) == EOF) {
        return false;
      }
      if (code.charAt(index) == '\\') {
        // escape
        index++;
      }
      index++;
    }
    index += 3;
    return true;
  }

  private void readStringPrefix(CodeReader code) {
    ch = Character.toUpperCase(code.charAt(index));
    if (PREFIX_CHARS.contains(ch)) {
      index++;
      ch = Character.toUpperCase(code.charAt(index));
    }
    if (PREFIX_CHARS.contains(ch)) {
      index++;
      ch = code.charAt(index);
    }
  }

  private static boolean isLookingOnLongString(CodeReader code, char ch, int index) {
    return (code.charAt(index) == ch) && (code.charAt(index + 1) == ch) && (code.charAt(index + 2) == ch);
  }

}
