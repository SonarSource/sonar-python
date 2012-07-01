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

import com.sonar.sslr.impl.Lexer;
import com.sonar.sslr.impl.channel.BlackHoleChannel;
import com.sonar.sslr.impl.channel.IdentifierAndKeywordChannel;
import com.sonar.sslr.impl.channel.PunctuatorChannel;
import com.sonar.sslr.impl.channel.UnknownCharacterChannel;
import org.sonar.python.PythonConfiguration;
import org.sonar.python.api.PythonKeyword;
import org.sonar.python.api.PythonPunctuator;
import org.sonar.python.api.PythonTokenType;

import static com.sonar.sslr.impl.channel.RegexpChannelBuilder.and;
import static com.sonar.sslr.impl.channel.RegexpChannelBuilder.commentRegexp;
import static com.sonar.sslr.impl.channel.RegexpChannelBuilder.o2n;
import static com.sonar.sslr.impl.channel.RegexpChannelBuilder.regexp;

public class PythonLexer {

  private PythonLexer() {
  }

  public static Lexer create() {
    return create(new PythonConfiguration());
  }

  private static final String EXP = "([Ee][+-]?+[0-9_]++)";

  public static Lexer create(PythonConfiguration conf) {
    LexerState lexerState = new LexerState();

    return Lexer.builder()
        .withCharset(conf.getCharset())
        .withFailIfNoChannelToConsumeOneCharacter(true)

        .withChannel(new NewLineChannel(lexerState))

        .withChannel(new IndentationChannel(lexerState))
        .withPreprocessor(new IndentationPreprocessor(lexerState))

        .withChannel(new BlackHoleChannel("\\s"))

        // http://docs.python.org/release/3.2/reference/lexical_analysis.html#comments
        .withChannel(commentRegexp("#[^\\n\\r]*+"))

        // http://docs.python.org/release/3.2/reference/lexical_analysis.html#string-and-bytes-literals
        // TODO 2.7 allows to use U"hello world" and UR"hello world"
        .withChannel(new LongStringLiteralsChannel())
        .withChannel(regexp(PythonTokenType.STRING, "(r|R)?+\'([^\'\\\\]*+(\\\\[\\s\\S])?+)*+\'"))
        .withChannel(regexp(PythonTokenType.STRING, "(r|R)?+\"([^\"\\\\]*+(\\\\[\\s\\S])?+)*+\""))

        // http://docs.python.org/release/3.2/reference/lexical_analysis.html#floating-point-literals
        .withChannel(regexp(PythonTokenType.NUMBER, "[0-9]++\\.[0-9]*+" + EXP + "?+"))
        .withChannel(regexp(PythonTokenType.NUMBER, "\\.[0-9]++" + EXP + "?+"))
        .withChannel(regexp(PythonTokenType.NUMBER, "[0-9]++" + EXP))

        // http://docs.python.org/release/3.2/reference/lexical_analysis.html#integer-literals
        // TODO 2.7 allows long integer literals, e.g. 3L
        .withChannel(regexp(PythonTokenType.NUMBER, "0[oO][0-7]++"))
        .withChannel(regexp(PythonTokenType.NUMBER, "0[xX][0-9a-fA-F]++"))
        .withChannel(regexp(PythonTokenType.NUMBER, "0[bB][01]++"))
        .withChannel(regexp(PythonTokenType.NUMBER, "[1-9][0-9]*+"))
        .withChannel(regexp(PythonTokenType.NUMBER, "0++"))

        // http://docs.python.org/release/3.2/reference/lexical_analysis.html#identifiers
        .withChannel(new IdentifierAndKeywordChannel(and("[a-zA-Z_]", o2n("\\w")), true, PythonKeyword.values()))

        // http://docs.python.org/release/3.2/reference/lexical_analysis.html#operators
        // http://docs.python.org/release/3.2/reference/lexical_analysis.html#delimiters
        .withChannel(new PunctuatorChannel(PythonPunctuator.values()))

        .withChannel(new UnknownCharacterChannel())

        .build();
  }
}
