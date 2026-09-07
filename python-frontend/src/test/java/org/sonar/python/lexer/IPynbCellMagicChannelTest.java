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

import com.sonar.sslr.api.GenericTokenType;
import com.sonar.sslr.api.Token;
import com.sonar.sslr.api.TokenType;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.sonar.python.api.PythonPunctuator;
import org.sonar.python.api.PythonTokenType;
import org.sonar.python.parser.IPythonParserConfiguration;
import org.sonar.python.parser.IPythonParserConfiguration.OpaqueCellRange;

import static java.util.stream.Collectors.joining;
import static org.assertj.core.api.Assertions.assertThat;

class IPynbCellMagicChannelTest {

  private static final String DELIMITER = "#SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER";

  @Test
  void singlePercentOpaqueCellShouldBeRawTokensUntilDelimiter() {
    var source = "%md\nIt's ordinary Markdown (\n" + DELIMITER + "\nanswer = 42\n";

    var tokens = lex(source, new OpaqueCellRange(1, 3));

    assertToken(tokens.get(0), PythonTokenType.IPYNB_CELL_MAGIC_PREFIX, "%", 1, 0);
    assertToken(tokens.get(1), PythonTokenType.IPYNB_CELL_MAGIC_BODY, "md\n", 1, 1);
    assertToken(tokens.get(2), PythonTokenType.IPYNB_CELL_MAGIC_BODY, "It's ordinary Markdown (\n", 2, 0);
    assertToken(tokens.get(3), PythonTokenType.IPYNB_CELL_DELIMITER, DELIMITER, 3, 0);
    assertThat(tokens.subList(0, 3).stream().map(Token::getValue).collect(joining()))
      .isEqualTo("%md\nIt's ordinary Markdown (\n");
    assertThat(tokens).anySatisfy(token -> assertToken(token, GenericTokenType.IDENTIFIER, "answer", 4, 0));
  }

  @Test
  void doublePercentOpaqueCellShouldKeepTwoModuloPrefixTokens() {
    var source = "%%sql\nSELECT '( FROM values\n" + DELIMITER + "\nanswer = 42\n";

    var tokens = lex(source, new OpaqueCellRange(1, 3));

    assertToken(tokens.get(0), PythonPunctuator.MOD, "%", 1, 0);
    assertToken(tokens.get(1), PythonPunctuator.MOD, "%", 1, 1);
    assertToken(tokens.get(2), PythonTokenType.IPYNB_CELL_MAGIC_BODY, "sql\n", 1, 2);
    assertToken(tokens.get(3), PythonTokenType.IPYNB_CELL_MAGIC_BODY, "SELECT '( FROM values\n", 2, 0);
    assertToken(tokens.get(4), PythonTokenType.IPYNB_CELL_DELIMITER, DELIMITER, 3, 0);
    assertThat(tokens).anySatisfy(token -> assertToken(token, GenericTokenType.IDENTIFIER, "answer", 4, 0));
  }

  private static List<Token> lex(String source, OpaqueCellRange range) {
    var lexerState = new LexerState();
    lexerState.reset();
    return PythonLexer.ipynbLexer(lexerState, new IPythonParserConfiguration(List.of(range))).lex(source);
  }

  private static void assertToken(Token token, TokenType type, String value, int line, int column) {
    assertThat(token.getType()).isEqualTo(type);
    assertThat(token.getValue()).isEqualTo(value);
    assertThat(token.getLine()).isEqualTo(line);
    assertThat(token.getColumn()).isEqualTo(column);
  }
}
