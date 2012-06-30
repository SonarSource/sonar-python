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

import com.sonar.sslr.api.GenericTokenType;
import com.sonar.sslr.impl.Lexer;
import org.junit.BeforeClass;
import org.junit.Test;
import org.sonar.python.api.PythonKeyword;
import org.sonar.python.api.PythonPunctuator;
import org.sonar.python.api.PythonTokenType;

import static com.sonar.sslr.test.lexer.LexerMatchers.hasComment;
import static com.sonar.sslr.test.lexer.LexerMatchers.hasToken;
import static org.hamcrest.Matchers.allOf;
import static org.hamcrest.Matchers.not;
import static org.junit.Assert.assertThat;

public class PythonLexerTest {

  private static Lexer lexer;

  @BeforeClass
  public static void init() {
    lexer = PythonLexer.create();
  }

  @Test
  public void comments() {
    assertThat(lexer.lex("# My comment \n new line"), hasComment("# My comment "));
  }

  @Test
  public void shortstring_literals() {
    assertThat("empty", lexer.lex("''"), hasToken("''", PythonTokenType.STRING));
    assertThat("empty", lexer.lex("\"\""), hasToken("\"\"", PythonTokenType.STRING));

    assertThat(lexer.lex("'hello world'"), hasToken("'hello world'", PythonTokenType.STRING));
    assertThat(lexer.lex("r'hello world'"), hasToken("r'hello world'", PythonTokenType.STRING));
    assertThat(lexer.lex("R'hello world'"), hasToken("R'hello world'", PythonTokenType.STRING));
    assertThat(lexer.lex("\"hello world\""), hasToken("\"hello world\"", PythonTokenType.STRING));
    assertThat(lexer.lex("r\"hello world\""), hasToken("r\"hello world\"", PythonTokenType.STRING));
    assertThat(lexer.lex("R\"hello world\""), hasToken("R\"hello world\"", PythonTokenType.STRING));

    assertThat("escaped single quote", lexer.lex("'\\''"), hasToken("'\\''", PythonTokenType.STRING));
    assertThat("escaped double quote", lexer.lex("\"\\\"\""), hasToken("\"\\\"\"", PythonTokenType.STRING));
  }

  @Test
  public void longstring_literals() {
    assertThat("multiline", lexer.lex("'''\n'''"), hasToken("'''\n'''", PythonTokenType.STRING));
    assertThat("multiline", lexer.lex("\"\"\"\n\"\"\""), hasToken("\"\"\"\n\"\"\"", PythonTokenType.STRING));

    assertThat("stringprefix", lexer.lex("r'''\n'''"), hasToken("r'''\n'''", PythonTokenType.STRING));
    assertThat("stringprefix", lexer.lex("R'''\n'''"), hasToken("R'''\n'''", PythonTokenType.STRING));
    assertThat("stringprefix", lexer.lex("r\"\"\"\n\"\"\""), hasToken("r\"\"\"\n\"\"\"", PythonTokenType.STRING));
    assertThat("stringprefix", lexer.lex("R\"\"\"\n\"\"\""), hasToken("R\"\"\"\n\"\"\"", PythonTokenType.STRING));

    assertThat("escaped single quote", lexer.lex("'''\\''''"), hasToken("'''\\''''", PythonTokenType.STRING));
    assertThat("escaped double quote", lexer.lex("\"\"\"\\\"\"\"\""), hasToken("\"\"\"\\\"\"\"\"", PythonTokenType.STRING));
  }

  @Test
  public void integer_literals() {
    assertThat(lexer.lex("7"), hasToken("7", PythonTokenType.NUMBER));
    assertThat(lexer.lex("0o177"), hasToken("0o177", PythonTokenType.NUMBER));
    assertThat(lexer.lex("0b100110111"), hasToken("0b100110111", PythonTokenType.NUMBER));
    assertThat(lexer.lex("0xdeadbeef"), hasToken("0xdeadbeef", PythonTokenType.NUMBER));
  }

  @Test
  public void floating_point_literals() {
    assertThat(lexer.lex("3.14"), hasToken("3.14", PythonTokenType.NUMBER));
    assertThat(lexer.lex("10."), hasToken("10.", PythonTokenType.NUMBER));
    assertThat(lexer.lex(".001"), hasToken(".001", PythonTokenType.NUMBER));
    assertThat(lexer.lex("1e100"), hasToken("1e100", PythonTokenType.NUMBER));
    assertThat(lexer.lex("3.14e-10"), hasToken("3.14e-10", PythonTokenType.NUMBER));
    assertThat(lexer.lex("0e0"), hasToken("0e0", PythonTokenType.NUMBER));
  }

  @Test
  public void identifiers() {
    assertThat(lexer.lex("True"), hasToken("True", PythonKeyword.TRUE));
    assertThat(lexer.lex("identifier"), hasToken("identifier", GenericTokenType.IDENTIFIER));
  }

  @Test
  public void operators_and_delimiters() {
    assertThat(lexer.lex("<<"), hasToken("<<", PythonPunctuator.LEFT_OP));
    assertThat(lexer.lex("+="), hasToken("+=", PythonPunctuator.PLUS_ASSIGN));
  }

  @Test
  public void indent_dedent() {
    assertThat(lexer.lex("    statement\n  statement"), allOf(hasToken("    ", PythonTokenType.INDENT), hasToken("  ", PythonTokenType.DEDENT)));
  }

  @Test
  public void implicit_line_joining() {
    assertThat(lexer.lex("month_names = ['January', \n 'December']"), not(hasToken("\n", PythonTokenType.NEWLINE)));
  }

  @Test
  public void explicit_line_joining() {
    assertThat(lexer.lex("line\r\nline"), hasToken(PythonTokenType.NEWLINE));
    assertThat(lexer.lex("line\rline"), hasToken(PythonTokenType.NEWLINE));
    assertThat(lexer.lex("line\nline"), hasToken(PythonTokenType.NEWLINE));

    assertThat(lexer.lex("line\\\r\nline"), not(hasToken(PythonTokenType.NEWLINE)));
    assertThat(lexer.lex("line\\\rline"), not(hasToken(PythonTokenType.NEWLINE)));
    assertThat(lexer.lex("line\\\nline"), not(hasToken(PythonTokenType.NEWLINE)));
  }

}
