/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2018 SonarSource SA
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
package org.sonar.python;

import com.google.common.base.Charsets;
import com.sonar.sslr.api.Token;
import com.sonar.sslr.impl.Lexer;
import java.util.List;
import org.junit.Test;
import org.sonar.python.lexer.PythonLexer;

import static org.assertj.core.api.Assertions.assertThat;

public class TokenLocationTest {

  private Lexer lexer = PythonLexer.create(new PythonConfiguration(Charsets.UTF_8));

  @Test
  public void test_multiline() throws Exception {
    TokenLocation tokenLocation = new TokenLocation(lex("'''first line\nsecond'''").get(0));
    assertOffsets(tokenLocation, 1, 0, 2, 9);

    tokenLocation = new TokenLocation(lex("'''first line\rsecond'''").get(0));
    assertOffsets(tokenLocation, 1, 0, 2, 9);

    tokenLocation = new TokenLocation(lex("'''first line\r\nsecond'''").get(0));
    assertOffsets(tokenLocation, 1, 0, 2, 9);
  }

  @Test
  public void test_newline_token() throws Exception {
    TokenLocation tokenLocation = new TokenLocation(lex("foo\n").get(1));
    assertOffsets(tokenLocation, 1, 3, 2, 0);
  }

  @Test
  public void test_one_line() throws Exception {
    TokenLocation tokenLocation = new TokenLocation(lex("  '''first line'''").get(1));
    assertOffsets(tokenLocation, 1, 2, 1, 18);

    tokenLocation = new TokenLocation(lex("foo").get(0));
    assertOffsets(tokenLocation, 1, 0, 1, 3);
  }

  @Test
  public void test_comment() throws Exception {
    TokenLocation commentLocation = new TokenLocation(lex("#comment\n").get(0).getTrivia().get(0).getToken());
    assertOffsets(commentLocation, 1, 0, 1, 8);
  }

  private void assertOffsets(TokenLocation tokenLocation, int startLine, int startLineOffset, int endLine, int endLineOffset) {
    assertThat(tokenLocation.startLine()).as("start line").isEqualTo(startLine);
    assertThat(tokenLocation.startLineOffset()).as("start line offset").isEqualTo(startLineOffset);
    assertThat(tokenLocation.endLine()).as("end line").isEqualTo(endLine);
    assertThat(tokenLocation.endLineOffset()).as("end line offset").isEqualTo(endLineOffset);
  }

  private List<Token> lex(String toLex) {
    return lexer.lex(toLex);
  }

}
