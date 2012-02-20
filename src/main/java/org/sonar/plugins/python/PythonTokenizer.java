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

package org.sonar.plugins.python;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.FileReader;

import net.sourceforge.pmd.cpd.SourceCode;
import net.sourceforge.pmd.cpd.TokenEntry;
import net.sourceforge.pmd.cpd.Tokenizer;
import net.sourceforge.pmd.cpd.Tokens;

import org.antlr.runtime.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.api.utils.SonarException;
import org.sonar.plugins.python.antlr.PythonLexer;

public class PythonTokenizer implements Tokenizer {

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

  public final void tokenize(SourceCode source, Tokens cpdTokens) {
    String fileName = source.getFileName();

    try {
      PythonLexer tokens = new MyLexer(new ANTLRFileStream(fileName));

      Token token = tokens.nextToken();
      while (token.getType() != Token.EOF) {
        if (token.getChannel() == Token.DEFAULT_CHANNEL
            && token.getType() != PythonLexer.LEADING_WS) {
          cpdTokens.add(new TokenEntry(token.getText(), fileName, token.getLine()));
        }
        token = tokens.nextToken();
      }
    } catch (Exception e) {
      String msg = new StringBuilder()
          .append("Cannot tokenize the file '")
          .append(fileName)
          .append("', details: '")
          .append(e)
          .append("'")
          .toString();
      throw new SonarException(msg, e);
    }

    cpdTokens.add(TokenEntry.getEOF());
  }
}
