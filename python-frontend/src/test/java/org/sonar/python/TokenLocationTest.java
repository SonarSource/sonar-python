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
package org.sonar.python;

import com.sonar.sslr.impl.Lexer;
import java.util.List;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.python.lexer.LexerState;
import org.sonar.python.lexer.PythonLexer;
import org.sonar.python.tree.TokenImpl;

import static org.assertj.core.api.Assertions.assertThat;

class TokenLocationTest {

  @Test
  void test_multiline() {
    TokenLocation tokenLocation = new TokenLocation(lex("'''first line\nsecond'''").get(0));
    assertOffsets(tokenLocation, 1, 0, 2, 9);

    tokenLocation = new TokenLocation(lex("'''first line\rsecond'''").get(0));
    assertOffsets(tokenLocation, 1, 0, 2, 9);

    tokenLocation = new TokenLocation(lex("'''first line\r\nsecond'''").get(0));
    assertOffsets(tokenLocation, 1, 0, 2, 9);
  }

  @Test
  void test_newline_token() {
    TokenLocation tokenLocation = new TokenLocation(lex("foo\n").get(1));
    assertOffsets(tokenLocation, 1, 3, 2, 0);
  }

  @Test
  void test_one_line() {
    TokenLocation tokenLocation = new TokenLocation(lex("  '''first line'''").get(1));
    assertOffsets(tokenLocation, 1, 2, 1, 18);

    tokenLocation = new TokenLocation(lex("foo").get(0));
    assertOffsets(tokenLocation, 1, 0, 1, 3);
  }

  @Test
  void test_comment() {
    TokenLocation commentLocation = new TokenLocation(lex("#comment\n").get(0).trivia().get(0).token());
    assertOffsets(commentLocation, 1, 0, 1, 8);

  }

  @Test
  void test_escaped_chars_ipython_lexer() {
    var token = new TokenImpl(iPythonLex("\"1\\n3\"").get(0), 3, 10, 3, List.of(), false);
    TokenLocation tokenLocation = new TokenLocation(token);
    assertOffsets(tokenLocation, 3, 10, 3, 19);

    token = new TokenImpl(iPythonLex("foo").get(0), 10, 20, 0, List.of(), false);
    tokenLocation = new TokenLocation(token);
    assertOffsets(tokenLocation, 10, 20, 10, 23);
  }

  @Test
  void test_multiline_ipython_lexer() {
    var tokens = iPythonLex("'''first line\nsecond\\t'''");
    var token = new TokenImpl(tokens.get(0), 3, 10, 1, List.of(), false);
    TokenLocation tokenLocation = new TokenLocation(token);
    assertOffsets(tokenLocation, 3, 10, 4, 11);
  }
  @Test
  void test_multiline_ipython_lexer_compressed() {
    var tokens = iPythonLex("'''first line\nsecond\\t'''");
    var token = new TokenImpl(tokens.get(0), 3, 10, 1, List.of(), true);
    TokenLocation tokenLocation = new TokenLocation(token);
    assertOffsets(tokenLocation, 3, 10, 3, 36);
  }
  private static void assertOffsets(TokenLocation tokenLocation, int startLine, int startLineOffset, int endLine, int endLineOffset) {
    assertThat(tokenLocation.startLine()).as("start line").isEqualTo(startLine);
    assertThat(tokenLocation.startLineOffset()).as("start line offset").isEqualTo(startLineOffset);
    assertThat(tokenLocation.endLine()).as("end line").isEqualTo(endLine);
    assertThat(tokenLocation.endLineOffset()).as("end line offset").isEqualTo(endLineOffset);
  }

  private List<com.sonar.sslr.api.Token> iPythonLex(String toLex) {
    LexerState lexerState = new LexerState();
    lexerState.reset();
    Lexer lexer = PythonLexer.create(lexerState);
    return lexer.lex(toLex);
  }

  private List<Token> lex(String toLex) {
    LexerState lexerState = new LexerState();
    lexerState.reset();
    Lexer lexer = PythonLexer.create(lexerState);
    return lexer.lex(toLex).stream().map(TokenImpl::new).collect(Collectors.toList());
  }

}
