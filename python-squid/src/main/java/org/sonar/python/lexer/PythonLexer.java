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

public final class PythonLexer {

  private static final String EXP = "([Ee][+-]?+[0-9_]++)";
  private static final String BYTES_PREFIX = "([bB][Rr]?|[rR][bB]?)";
  private static final String IMAGINARY_SUFFIX = "(j|J)";
  private static final String LONG_INTEGER_SUFFIX = "(l|L)";
  private static final String FORMATTED_STRING_PREFIX = "([fF][rR]?|[rR][fF]?)";

  private PythonLexer() {
  }

  public static Lexer create(PythonConfiguration conf) {
    LexerState lexerState = new LexerState();

    return Lexer.builder()
        .withCharset(conf.getCharset())
        .withFailIfNoChannelToConsumeOneCharacter(true)

        .withChannel(new NewLineChannel(lexerState))

        .withChannel(new IndentationChannel(lexerState))
        .withPreprocessor(new IndentationPreprocessor(lexerState))

        .withChannel(new BlackHoleChannel("\\s"))

        // http://docs.python.org/reference/lexical_analysis.html#comments
        .withChannel(commentRegexp("#[^\\n\\r]*+"))

        // http://docs.python.org/reference/lexical_analysis.html#string-literals
        .withChannel(new StringLiteralsChannel())

        // http://docs.python.org/release/3.2/reference/lexical_analysis.html#string-and-bytes-literals
        .withChannel(regexp(PythonTokenType.STRING, BYTES_PREFIX + "\'([^\'\\\\]*+(\\\\[\\s\\S])?+)*+\'"))
        .withChannel(regexp(PythonTokenType.STRING, BYTES_PREFIX + "\"([^\"\\\\]*+(\\\\[\\s\\S])?+)*+\""))

        //https://docs.python.org/3.6/reference/lexical_analysis.html#formatted-string-literals
      .withChannel(regexp(PythonTokenType.STRING, FORMATTED_STRING_PREFIX + "\'([^\'\\\\]*+(\\\\[\\s\\S])?+)*+\'"))
      .withChannel(regexp(PythonTokenType.STRING, FORMATTED_STRING_PREFIX + "\"([^\"\\\\]*+(\\\\[\\s\\S])?+)*+\""))

        // http://docs.python.org/reference/lexical_analysis.html#floating-point-literals
        // http://docs.python.org/reference/lexical_analysis.html#imaginary-literals
        .withChannel(regexp(PythonTokenType.NUMBER, "[0-9]++\\.[0-9]*+" + EXP + "?+" + IMAGINARY_SUFFIX + "?+"))
        .withChannel(regexp(PythonTokenType.NUMBER, "\\.[0-9]++" + EXP + "?+" + IMAGINARY_SUFFIX + "?+"))
        .withChannel(regexp(PythonTokenType.NUMBER, "[0-9]++" + EXP + IMAGINARY_SUFFIX + "?+"))
        .withChannel(regexp(PythonTokenType.NUMBER, "[0-9]++" + IMAGINARY_SUFFIX))

        // http://docs.python.org/reference/lexical_analysis.html#integer-and-long-integer-literals
        .withChannel(regexp(PythonTokenType.NUMBER, "0[oO]?+[0-7]++" + LONG_INTEGER_SUFFIX + "?+"))
        .withChannel(regexp(PythonTokenType.NUMBER, "0[xX][0-9a-fA-F]++" + LONG_INTEGER_SUFFIX + "?+"))
        .withChannel(regexp(PythonTokenType.NUMBER, "0[bB][01]++" + LONG_INTEGER_SUFFIX + "?+"))
        .withChannel(regexp(PythonTokenType.NUMBER, "[1-9][0-9]*+" + LONG_INTEGER_SUFFIX + "?+"))
        .withChannel(regexp(PythonTokenType.NUMBER, "0++" + LONG_INTEGER_SUFFIX + "?+"))

        // http://docs.python.org/reference/lexical_analysis.html#identifiers
        .withChannel(new IdentifierAndKeywordChannel(and("[a-zA-Z_]", o2n("\\w")), true, PythonKeyword.values()))

        // http://docs.python.org/reference/lexical_analysis.html#operators
        // http://docs.python.org/reference/lexical_analysis.html#delimiters
        .withChannel(new PunctuatorChannel(PythonPunctuator.values()))

        .withChannel(new UnknownCharacterChannel())

        .build();
  }
}
