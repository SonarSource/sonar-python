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
import com.sonar.sslr.api.Grammar;
import com.sonar.sslr.api.Rule;
import com.sonar.sslr.api.Token;
import com.sonar.sslr.impl.Lexer;
import com.sonar.sslr.impl.Parser;
import com.sonar.sslr.impl.matcher.RuleDefinition;
import java.util.ArrayList;
import java.util.List;
import org.sonar.python.api.IPythonGrammarBuilder;
import org.sonar.python.api.PythonGrammarBuilder;
import org.sonar.python.api.PythonTokenType;
import org.sonar.python.lexer.LexerState;
import org.sonar.python.lexer.PythonLexer;

public final class PythonParser {

  private final Parser<Grammar> sslrParser;

  public static PythonParser create() {
    return new PythonParser(new PythonGrammarBuilder().create());
  }

  public static PythonParser createIPythonParser() {
    return new PythonParser(new IPythonGrammarBuilder().create());
  }

  private PythonParser(Grammar grammar) {
    sslrParser = new SslrPythonParser(grammar);
  }

  public AstNode parse(String source) {
    return sslrParser.parse(source);
  }

  public void setRootRule(Rule rule) {
    sslrParser.setRootRule(rule);
  }

  public Grammar getGrammar() {
    return sslrParser.getGrammar();
  }

  public RuleDefinition getRootRule() {
    return sslrParser.getRootRule();
  }

  // We can't use com.sonar.sslr.impl.Parser directly because we need to add
  // DEDENT tokens before the EOF token (without using SSLR deprecated preprocessor API)
  // and we can't create a subclass of com.sonar.sslr.impl.Lexer.
  // The only solution seems to subclass com.sonar.sslr.impl.Parser.
  private static class SslrPythonParser extends Parser<Grammar> {
    private final LexerState lexerState;
    private final Lexer lexer;

    private SslrPythonParser(Grammar grammar) {
      super(grammar);
      super.setRootRule(super.getGrammar().getRootRule());
      this.lexerState = new LexerState();
      this.lexer = PythonLexer.create(lexerState);
    }

    @Override
    public AstNode parse(String source) {
      lexerState.reset();
      lexer.lex(source);
      List<Token> tokens = tokens();
      return super.parse(tokens);
    }

    private List<Token> tokens() {
      List<Token> tokens = lexer.getTokens();
      if (lexerState.indentationStack.peek() > 0) {
        Token eofToken = tokens.get(tokens.size() - 1);
        tokens = new ArrayList<>(tokens.subList(0, tokens.size() - 1));
        while (lexerState.indentationStack.peek() > 0) {
          lexerState.indentationStack.pop();
          tokens.add(Token.builder()
            .setURI(eofToken.getURI())
            .setType(PythonTokenType.DEDENT)
            .setLine(eofToken.getLine())
            .setColumn(eofToken.getColumn())
            .setValueAndOriginalValue("")
            .build());
        }
        tokens.add(eofToken);
      }
      return tokens;
    }
  }

}
