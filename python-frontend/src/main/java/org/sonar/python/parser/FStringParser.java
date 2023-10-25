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
import com.sonar.sslr.api.Token;
import com.sonar.sslr.impl.Lexer;
import com.sonar.sslr.impl.Parser;
import java.util.List;
import org.sonar.python.api.PythonGrammarBuilder;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.lexer.LexerState;
import org.sonar.python.lexer.PythonLexer;
import org.sonar.python.tree.StringElementImpl;
import org.sonar.python.tree.TokenImpl;

public class FStringParser {

  private final LexerState lexerState;
  private final Lexer lexer;
  private final Parser<Grammar> internalParser = Parser.builder(new PythonGrammarBuilder().create()).build();

  public FStringParser() {
    this.lexerState = new LexerState();
    this.lexer = PythonLexer.fStringLexer(lexerState);
    this.internalParser.setRootRule(internalParser.getGrammar().rule(PythonGrammar.FSTRING));
  }

  public List<AstNode> fStringExpressions(Token fStringToken) {
    StringElementImpl element = new StringElementImpl(new TokenImpl(fStringToken));
    String literalValue = element.trimmedQuotesValue();
    lexerState.reset(fStringToken.getLine(), fStringToken.getColumn() + element.contentStartIndex());
    lexer.lex(literalValue);
    List<Token> tokens = lexer.getTokens();
    AstNode astNode = internalParser.parse(tokens);
    return astNode.getChildren(PythonGrammar.FORMATTED_EXPR);
  }

}
