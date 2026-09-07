/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
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

import com.sonar.sslr.api.Token;
import com.sonar.sslr.api.TokenType;
import com.sonar.sslr.impl.Lexer;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;
import org.sonar.python.api.PythonPunctuator;
import org.sonar.python.api.PythonTokenType;
import org.sonar.python.parser.IPythonParserConfiguration.OpaqueCellRange;
import org.sonar.sslr.channel.Channel;
import org.sonar.sslr.channel.CodeReader;

public class IPynbCellMagicChannel extends Channel<Lexer> {

  private final Map<Integer, OpaqueCellRange> opaqueCellsByStartLine;

  public IPynbCellMagicChannel(List<OpaqueCellRange> opaqueCellRanges) {
    this.opaqueCellsByStartLine = opaqueCellRanges.stream()
      .collect(Collectors.toUnmodifiableMap(OpaqueCellRange::startLine, Function.identity()));
  }

  @Override
  public boolean consume(CodeReader code, Lexer lexer) {
    int line = code.getLinePosition();
    int column = code.getColumnPosition();
    OpaqueCellRange opaqueCell = opaqueCellsByStartLine.get(line);
    if (column != 0 || code.peek() != '%' || opaqueCell == null) {
      return false;
    }

    consumePrefix(code, lexer);
    consumeOpaqueBody(code, lexer, opaqueCell.delimiterLine());
    return true;
  }

  private static void consumePrefix(CodeReader code, Lexer lexer) {
    boolean doublePercentPrefix = code.length() > 1 && code.charAt(1) == '%';
    addToken(code, lexer, doublePercentPrefix ? PythonPunctuator.MOD : PythonTokenType.IPYNB_CELL_MAGIC_PREFIX, "%");
    if (code.peek() == '%') {
      addToken(code, lexer, PythonPunctuator.MOD, "%");
    }
  }

  private static void consumeOpaqueBody(CodeReader code, Lexer lexer, int delimiterLine) {
    while (code.getLinePosition() < delimiterLine && code.peek() != -1) {
      int line = code.getLinePosition();
      int column = code.getColumnPosition();
      StringBuilder value = new StringBuilder();
      do {
        code.pop(value);
      } while (code.getLinePosition() == line && code.peek() != -1);
      lexer.addToken(Token.builder()
        .setType(PythonTokenType.IPYNB_CELL_MAGIC_BODY)
        .setValueAndOriginalValue(value.toString())
        .setURI(lexer.getURI())
        .setLine(line)
        .setColumn(column)
        .build());
    }
  }

  private static void addToken(CodeReader code, Lexer lexer, TokenType type, String value) {
    int line = code.getLinePosition();
    int column = code.getColumnPosition();
    code.pop();
    lexer.addToken(Token.builder()
      .setType(type)
      .setValueAndOriginalValue(value)
      .setURI(lexer.getURI())
      .setLine(line)
      .setColumn(column)
      .build());
  }
}
