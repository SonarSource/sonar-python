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
package org.sonar.python.tree;

import com.sonar.sslr.api.Token;
import com.sonar.sslr.impl.Lexer;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.sonar.python.IPythonLocation;
import org.sonar.python.lexer.LexerState;
import org.sonar.python.lexer.PythonLexer;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.sonar.python.PythonTestUtils.mapToColumnMappingList;

class TokenEnricherTest {
  private static TestLexer lexer;

  @BeforeAll
  static void init() {
    lexer = new TestLexer();
  }

  private static class TestLexer {
    private LexerState lexerState = new LexerState();
    private Lexer lexer = PythonLexer.create(lexerState);

    List<Token> lex(String code) {
      lexerState.reset();
      return lexer.lex(code);
    }
  }

  @Test
  void shouldReturnAnEmptyList() {
    // when no tokens are provided
    var tokens = TokenEnricher.enrichTokens(List.of(), Map.of());
    assertThat(tokens).isEmpty();
  }

  @Test
  void shouldNotModifyTheTokens() {
    // when there are no offset
    var code = "a = 1";

    var originalTokens = lexer.lex(code);
    var tokens = TokenEnricher.enrichTokens(originalTokens, Map.of());
    var lines = tokens.stream().map(TokenImpl::line).toList();
    assertThat(lines).containsExactlyElementsOf(originalTokens.stream().map(Token::getLine).toList());
    var columns = tokens.stream().map(TokenImpl::column).toList();
    assertThat(columns).containsExactlyElementsOf(originalTokens.stream().map(Token::getColumn).toList());
    var countedEscaped = tokens.stream().map(TokenImpl::includedEscapeChars).toList();
    assertThat(countedEscaped).containsOnly(0);
  }

  @Test
  void shouldThrowIllegalStateException() {
    //when the mapping is not present for the current line
    var code = "a = 1\n\nb=3";
    var offsetMap = Map.of(
      1, new IPythonLocation(200, 23),
      2, new IPythonLocation(201, 23));
    var originalTokens = lexer.lex(code);
    Throwable throwable = assertThrows(IllegalStateException.class, () -> TokenEnricher.enrichTokens(originalTokens, offsetMap));
    assertThat(throwable.getMessage()).isEqualTo("No IPythonLocation found for line 3");
  }

  @Test
  void shouldProvideOffsetForEscapeChar() {
    var code = "a = \"1\"";
    var expectedTokens = lexer.lex(code);
    var escapedChars = mapToColumnMappingList(Map.of(4, 1, 6, 1));
    var tokens = TokenEnricher.enrichTokens(expectedTokens, Map.of(1, new IPythonLocation(100, 300, escapedChars)));
    var stringToken = tokens.get(2);
    assertThat(stringToken.line()).isEqualTo(100);
    assertThat(stringToken.column()).isEqualTo(304);
    assertThat(stringToken.includedEscapeChars()).isEqualTo(2);

    var eofToken = tokens.get(3);
    assertThat(eofToken.line()).isEqualTo(100);
    assertThat(eofToken.column()).isEqualTo(309);
  }

  @Test
  void shouldComputeColCorrectly() {
    var code = "a = f\"{b} \\n test\" + \"1\"";
    var expectedTokens = lexer.lex(code);
    var escapedChars = mapToColumnMappingList(Map.ofEntries(
      Map.entry(5, 1),
      Map.entry(10, 1),
      Map.entry(17, 1),
      Map.entry(21, 1),
      Map.entry(23, 1)
    ));
    var tokens = TokenEnricher.enrichTokens(expectedTokens, Map.of(1, new IPythonLocation(100, 300, escapedChars)));
    var stringToken = tokens.get(tokens.size() - 2);
    assertThat(stringToken.line()).isEqualTo(100);
    assertThat(stringToken.column()).isEqualTo(324);
    assertThat(stringToken.includedEscapeChars()).isEqualTo(2);

    var eofToken = tokens.get(tokens.size() - 1);
    assertThat(eofToken.line()).isEqualTo(100);
    assertThat(eofToken.column()).isEqualTo(329);
    assertThat(eofToken.includedEscapeChars()).isZero();
  }

  @Test
  void shouldComputeTabColCorrectly() {
    var code = "\ta";
    var expectedTokens = lexer.lex(code);
    var escapedChars = mapToColumnMappingList(Map.of(0, 1));
    var tokens = TokenEnricher.enrichTokens(expectedTokens, Map.of(1, new IPythonLocation(100, 300, escapedChars)));
    var tabToken = tokens.get(0);
    assertThat(tabToken.line()).isEqualTo(100);
    assertThat(tabToken.column()).isEqualTo(300);
    assertThat(tabToken.includedEscapeChars()).isEqualTo(1);

    var idToken = tokens.get(1);
    assertThat(idToken.line()).isEqualTo(100);
    assertThat(idToken.column()).isEqualTo(302);
    assertThat(idToken.includedEscapeChars()).isZero();

    var eofToken = tokens.get(2);
    assertThat(eofToken.line()).isEqualTo(100);
    assertThat(eofToken.column()).isEqualTo(303);
    assertThat(eofToken.includedEscapeChars()).isZero();
  }

  @Test
  void shouldComputeColCorrectlyForTrivia() {
    var code = "a = 3 # comment";
    var expectedTokens = lexer.lex(code);
    var tokens = TokenEnricher.enrichTokens(expectedTokens, Map.of(1, new IPythonLocation(100, 300)));
    var trivias = tokens.get(tokens.size() - 1).trivia();
    assertThat(trivias).hasSize(1);
    assertThat(trivias.get(0).token().line()).isEqualTo(100);
    assertThat(trivias.get(0).token().column()).isEqualTo(306);
    assertThat(trivias.get(0).token().includedEscapeChars()).isZero();
  }

  @Test
  void shouldComputeColCorrectlyForTriviaWithEscapeChar() {
    var code = "a = 3 # test\\n";
    var expectedTokens = lexer.lex(code);
    var escapedChars = mapToColumnMappingList(Map.of(12, 1));
    var tokens = TokenEnricher.enrichTokens(expectedTokens, Map.of(1, new IPythonLocation(100, 300, escapedChars)));
    var trivias = tokens.get(tokens.size() - 1).trivia();
    assertThat(trivias).hasSize(1);
    assertThat(trivias.get(0).token().line()).isEqualTo(100);
    assertThat(trivias.get(0).token().column()).isEqualTo(306);
    assertThat(trivias.get(0).token().includedEscapeChars()).isEqualTo(1);
  }

  @Test
  void shouldComputeColCorrectlyForTriviaOnDifferentLine() {
    var code = "# comment\na = 3";
    var expectedTokens = lexer.lex(code);
    var tokens = TokenEnricher.enrichTokens(expectedTokens, Map.of(1, new IPythonLocation(100, 300), 2,
      new IPythonLocation(101, 300)));
    assertThat(tokens.get(0).line()).isEqualTo(101);
    var trivias = tokens.get(0).trivia();
    assertThat(trivias).hasSize(1);
    assertThat(trivias.get(0).token().line()).isEqualTo(100);
    assertThat(trivias.get(0).token().column()).isEqualTo(300);
    assertThat(trivias.get(0).token().includedEscapeChars()).isZero();
  }

  @Test
  void shouldComputeCorrectlyForSingleQuote() {
    var code = "a = '1'";
    var expectedTokens = lexer.lex(code);
    var tokens = TokenEnricher.enrichTokens(expectedTokens, Map.of(1, new IPythonLocation(100, 300)));
    var stringToken = tokens.get(2);
    assertThat(stringToken.line()).isEqualTo(100);
    assertThat(stringToken.column()).isEqualTo(304);
    assertThat(stringToken.includedEscapeChars()).isZero();
  }
}
