/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2023 SonarSource SA
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
package org.sonar.python.parser;

import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.Token;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.sonar.python.api.PythonTokenType;

import static org.assertj.core.api.Assertions.assertThat;

class FStringParserTest {

  private final FStringParser parser = new FStringParser();

  @Test
  void empty_f_string() {
    assertThat(parse("f''")).isEmpty();
    assertThat(parse("f\"\"")).isEmpty();
    assertThat(parse("f''''''")).isEmpty();
  }

  @Test
  void no_formatted_expression() {
    assertThat(parse("f'a'")).isEmpty();
    assertThat(parse("f\"a\"")).isEmpty();
    assertThat(parse("f'''a'''")).isEmpty();
  }

  @Test
  void name_expression() {
    List<AstNode> nodes = parse("f'hello {var}!'");
    assertThat(nodes).hasSize(1);
    assertThat(nodes.get(0).getTokens()).extracting(Token::getValue).containsExactly("{", "var", "}");
  }

  @Test
  void triple_quoted() {
    List<AstNode> nodes = parse("f'''hello '{var}'!'''");
    assertThat(nodes).hasSize(1);
    assertThat(nodes.get(0).getTokens()).extracting(Token::getValue).containsExactly("{", "var", "}");
  }

  @Test
  void escaped_curly_brace() {
    assertThat(parse("f'{{abc}}'")).isEmpty();
    assertThat(parse("f'{{abc}}{xyz}'").get(0).getTokens()).extracting(Token::getValue).containsExactly("{", "xyz", "}");
  }

  @Test
  void expressions_should_not_be_merged() {
    assertThat(parse("f'{x} {+y}!'")).hasSize(2);
  }

  @Test
  void token_line_and_column() {
    Token varToken = parse("f'hello {var}!'", 42, 5).get(0).getTokens().get(1);
    assertThat(varToken.getValue()).isEqualTo("var");
    assertThat(varToken.getLine()).isEqualTo(42);
    assertThat(varToken.getColumn()).isEqualTo(14);
  }

  @Test
  void token_line_and_column_in_multiline_f_string() {
    Token varToken = parse("f'''hello\n {var}'''", 42, 5).get(0).getTokens().get(1);
    assertThat(varToken.getValue()).isEqualTo("var");
    assertThat(varToken.getLine()).isEqualTo(43);
    assertThat(varToken.getColumn()).isEqualTo(2);
  }

  @Test
  void conversions() {
    assertThat(parse("f'{x!a}'")).hasSize(1);
    assertThat(parse("f'{foo(\"!a\")!a}'")).hasSize(1);
    assertThat(parse("f'{user=!s}'")).hasSize(1);
  }

  @Test
  void format_specifiers() {
    assertThat(parse("f'{today:%B %d, %Y}'")).hasSize(1);
    assertThat(parse("f'{number:#0x}'")).hasSize(1);
    assertThat(parse("f'result: {value:{width}.{precision}}'")).hasSize(1);
    assertThat(parse("f'{delta.days=:,d}'")).hasSize(1);
  }

  private List<AstNode> parse(String tokenValue) {
    return parse(tokenValue, 1, 1);
  }

  private List<AstNode> parse(String tokenValue, int line, int column) {
    Token token = Token.builder()
      .setLine(line)
      .setColumn(column)
      .setValueAndOriginalValue(tokenValue)
      .setType(PythonTokenType.STRING)
      .setURI(fakeUri())
      .build();
    return parser.fStringExpressions(token);
  }

  private URI fakeUri() {
    try {
      return new URI("tests://unittest");
    } catch (URISyntaxException e) {
      throw new RuntimeException(e);
    }
  }
}
