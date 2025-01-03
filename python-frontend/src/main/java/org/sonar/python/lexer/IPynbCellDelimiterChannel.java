/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
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

import com.sonar.sslr.api.Token;
import com.sonar.sslr.impl.Lexer;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.sonar.python.api.PythonTokenType;
import org.sonar.sslr.channel.Channel;
import org.sonar.sslr.channel.CodeReader;

public class IPynbCellDelimiterChannel extends Channel<Lexer> {

  private final StringBuilder tmpBuilder = new StringBuilder();
  private final Matcher sonarLintVSCodeCellDelimiter = Pattern.compile("#SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER").matcher("");
  private final Matcher cellDelimiter = Pattern.compile("#\\h?%%(\\h+\\w+)?").matcher("");
  private final Token.Builder tokenBuilder = Token.builder();
  private final LexerState lexerState;

  public IPynbCellDelimiterChannel(LexerState lexerState) {
    this.lexerState = lexerState;
  }

  @Override
  public boolean consume(CodeReader code, Lexer lexer) {
    if (code.getColumnPosition() != 0) {
      // Cell delimiters must be at the beginning of a line
      return false;
    }
    if (code.popTo(sonarLintVSCodeCellDelimiter, tmpBuilder) > 0 || code.popTo(cellDelimiter, tmpBuilder) > 0) {
      resetIndentationLevel(code, lexer);
      String value = tmpBuilder.toString();

      Token token = tokenBuilder
        .setType(PythonTokenType.IPYNB_CELL_DELIMITER)
        .setValueAndOriginalValue(value)
        .setURI(lexer.getURI())
        .setLine(code.getPreviousCursor().getLine())
        .setColumn(code.getPreviousCursor().getColumn())
        .build();

      lexer.addToken(token);

      tmpBuilder.delete(0, tmpBuilder.length());
      return true;
    }
    return false;
  }

  private void resetIndentationLevel(CodeReader code, Lexer lexer) {
    while (lexerState.indentationStack.peek() > 0) {
      lexerState.indentationStack.pop();
      lexer.addToken(Token.builder()
        .setType(PythonTokenType.DEDENT)
        .setValueAndOriginalValue("")
        .setURI(lexer.getURI())
        .setLine(code.getLinePosition())
        .setColumn(0)
        .build());
    }
  }
}
