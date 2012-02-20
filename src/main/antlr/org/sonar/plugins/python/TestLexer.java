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

import org.antlr.runtime.*;
import java.io.IOException;

class TestLexer {

  // override nextToken to set startPos
  public static class MyLexer extends PythonLexer {

    public MyLexer(CharStream lexer) {
      super(lexer);
    }

    public Token nextToken() {
      startPos = getCharPositionInLine();
      return super.nextToken();
    }
  }

  public static void main(String[] args) throws IOException {
    // MyLexer lexer = new MyLexer(new ANTLRFileStream(args[0]));
    // PythonTokenSource tokens =
    // new PythonTokenSource(new CommonTokenStream(lexer));

    // Or just work with undecorated lexer. The difference to the above
    // are the LEADING_WS-Tokens, which are in the above variant consumed
    // by the decorator and turned into virtual indent/dedent tokens
    MyLexer tokens = new MyLexer(new ANTLRFileStream(args[0]));

    Token token = tokens.nextToken();
    while (token.getType() != Token.EOF) {
      if (token.getChannel() == Token.DEFAULT_CHANNEL
          && token.getType() != PythonLexer.NEWLINE)
        System.out.println("'" + token.getText() + "' " + token.getType() + " on channel: " + token.getChannel());
      token = tokens.nextToken();
    }
  }
}
