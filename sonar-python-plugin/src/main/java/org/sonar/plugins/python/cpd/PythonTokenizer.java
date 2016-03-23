/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2016 SonarSource SA
 * mailto:contact AT sonarsource DOT com
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
package org.sonar.plugins.python.cpd;

import com.sonar.sslr.api.Token;
import com.sonar.sslr.impl.Lexer;
import net.sourceforge.pmd.cpd.SourceCode;
import net.sourceforge.pmd.cpd.TokenEntry;
import net.sourceforge.pmd.cpd.Tokenizer;
import net.sourceforge.pmd.cpd.Tokens;
import org.sonar.python.PythonConfiguration;
import org.sonar.python.api.PythonTokenType;
import org.sonar.python.lexer.PythonLexer;

import java.io.File;
import java.nio.charset.Charset;
import java.util.List;

public class PythonTokenizer implements Tokenizer {

  private final Charset charset;

  public PythonTokenizer(Charset charset) {
    this.charset = charset;
  }

  @Override
  public final void tokenize(SourceCode source, Tokens cpdTokens) {
    Lexer lexer = PythonLexer.create(new PythonConfiguration(charset));
    String fileName = source.getFileName();
    List<Token> tokens = lexer.lex(new File(fileName));
    for (Token token : tokens) {
      if (!token.getType().equals(PythonTokenType.NEWLINE) && !token.getType().equals(PythonTokenType.DEDENT) && !token.getType().equals(PythonTokenType.INDENT)) {
        TokenEntry cpdToken = new TokenEntry(getTokenImage(token), fileName, token.getLine());
        cpdTokens.add(cpdToken);
      }
    }
    cpdTokens.add(TokenEntry.getEOF());
  }

  private String getTokenImage(Token token) {
    return token.getValue();
  }

}
