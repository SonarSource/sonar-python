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
import java.util.List;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.sonar.python.api.PythonTokenType;

import static com.sonar.sslr.test.lexer.LexerMatchers.hasToken;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.not;

class IPythonLexerTest {
  private static TestLexer lexer;

  @BeforeAll
  static void init() {
    lexer = new IPythonLexerTest.TestLexer();
  }

  private static class TestLexer {
    private LexerState lexerState = new LexerState();
    private Lexer lexer = PythonLexer.ipynbLexer(lexerState);

    List<Token> lex(String code) {
      lexerState.reset();
      return lexer.lex(code);
    }
  }

  @Test
  void sonarLintVSCodeCellDelimiterTest() {
    assertThat(lexer.lex("foo\n#SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER"), hasToken(PythonTokenType.IPYNB_CELL_DELIMITER));
    assertThat(lexer.lex("foo #SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER"), not(hasToken(PythonTokenType.IPYNB_CELL_DELIMITER)));
    assertThat(lexer.lex("if cond:\n  foo()\n#SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER"), hasToken(PythonTokenType.DEDENT));
  }

  @Test
  void cellDelimiterTest() {
    assertThat(lexer.lex("#%%\ndef foo():\n    pass\n#%%"), hasToken(PythonTokenType.IPYNB_CELL_DELIMITER));
    assertThat(lexer.lex("#%% md\nthis is text ="), hasToken(PythonTokenType.IPYNB_CELL_DELIMITER));
    assertThat(lexer.lex("foo\n#%%"), hasToken(PythonTokenType.IPYNB_CELL_DELIMITER));
    assertThat(lexer.lex("foo\n# %%"), hasToken(PythonTokenType.IPYNB_CELL_DELIMITER));
    assertThat(lexer.lex("foo #%% md"), not(hasToken(PythonTokenType.IPYNB_CELL_DELIMITER)));
    assertThat(lexer.lex("if cond:\n  foo()\n# %% [markdown]"), hasToken(PythonTokenType.DEDENT));
  }
}
