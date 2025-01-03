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

import com.sonar.sslr.impl.Lexer;
import com.sonar.sslr.impl.channel.BlackHoleChannel;
import com.sonar.sslr.impl.channel.IdentifierAndKeywordChannel;
import com.sonar.sslr.impl.channel.PunctuatorChannel;
import com.sonar.sslr.impl.channel.UnknownCharacterChannel;
import org.sonar.python.api.PythonKeyword;
import org.sonar.python.api.PythonPunctuator;
import org.sonar.python.api.PythonTokenType;

import static com.sonar.sslr.impl.channel.RegexpChannelBuilder.and;
import static com.sonar.sslr.impl.channel.RegexpChannelBuilder.or;
import static com.sonar.sslr.impl.channel.RegexpChannelBuilder.commentRegexp;
import static com.sonar.sslr.impl.channel.RegexpChannelBuilder.o2n;
import static com.sonar.sslr.impl.channel.RegexpChannelBuilder.regexp;

public final class PythonLexer {

  private static final String EXP = "([Ee][+-]?+[0-9_]++)";
  private static final String BYTES_PREFIX = "([bB][Rr]?|[rR][bB]?)";
  private static final String IMAGINARY_SUFFIX = "(j|J)";
  private static final String LONG_INTEGER_SUFFIX = "(l|L)";
  private static final String UNICODE_CHAR = "[^\u0000-\u007F]";
  private static final String IDENTIFIER_START = "[\\p{Lu}\\p{Ll}\\p{Lt}\\p{Lm}\\p{Lo}\\p{Nl}_]";
  private static final String IDENTIFIER_CONTINUE = "[" + IDENTIFIER_START + "\\p{Mn}\\p{Mc}\\p{Nd}\\p{Pc}]";

  private static final String SINGLE_QUOTE_STRING = "\'([^\'\\\\]*+(\\\\[\\s\\S])?+)*+\'";
  private static final String DOUBLE_QUOTES_STRING = "\"([^\"\\\\]*+(\\\\[\\s\\S])?+)*+\"";

  private static final String NUMBER_REGEX = "[0-9]++(_?[0-9])*+";

  private PythonLexer() {
  }

  public static Lexer create(LexerState lexerState) {
    Lexer.Builder builder = Lexer.builder().withFailIfNoChannelToConsumeOneCharacter(true);
    addCommonChannels(builder, lexerState);
    return builder.build();
  }

  public static Lexer ipynbLexer(LexerState lexerState) {
    Lexer.Builder builder = Lexer.builder().withFailIfNoChannelToConsumeOneCharacter(true);
    builder.withChannel(new IPynbCellDelimiterChannel(lexerState));
    addCommonChannels(builder, lexerState);
    return builder.build();
  }

  private static void addCommonChannels(Lexer.Builder builder, LexerState lexerState) {
    builder
      .withChannel(new NewLineChannel(lexerState))

      .withChannel(new IndentationChannel(lexerState))

      .withChannel(new BlackHoleChannel("\\s"))

      // http://docs.python.org/reference/lexical_analysis.html#comments
      .withChannel(commentRegexp("#[^\\n\\r]*+"))

      // http://docs.python.org/reference/lexical_analysis.html#string-literals
      .withChannel(new StringLiteralsChannel())

      //https://docs.python.org/3.6/reference/lexical_analysis.html#formatted-string-literals
      .withChannel(new FStringChannel(lexerState))

      // http://docs.python.org/release/3.2/reference/lexical_analysis.html#string-and-bytes-literals
      .withChannel(regexp(PythonTokenType.STRING, BYTES_PREFIX + SINGLE_QUOTE_STRING))
      .withChannel(regexp(PythonTokenType.STRING, BYTES_PREFIX + DOUBLE_QUOTES_STRING))

      // http://docs.python.org/reference/lexical_analysis.html#floating-point-literals
      // http://docs.python.org/reference/lexical_analysis.html#imaginary-literals
      // https://www.python.org/dev/peps/pep-0515/
      .withChannel(regexp(PythonTokenType.NUMBER, "[0-9]++(_?[0-9])*+\\.[0-9]*+(_?[0-9])*+" + EXP + "?+" + IMAGINARY_SUFFIX + "?+"))
      .withChannel(regexp(PythonTokenType.NUMBER, "\\.[0-9]++(_?[0-9])*+" + EXP + "?+" + IMAGINARY_SUFFIX + "?+"))
      .withChannel(regexp(PythonTokenType.NUMBER, NUMBER_REGEX + EXP + IMAGINARY_SUFFIX + "?+"))
      .withChannel(regexp(PythonTokenType.NUMBER, NUMBER_REGEX + IMAGINARY_SUFFIX))

      // http://docs.python.org/reference/lexical_analysis.html#integer-and-long-integer-literals
      // https://www.python.org/dev/peps/pep-0515/
      .withChannel(regexp(PythonTokenType.NUMBER, "0[oO]?+(_?[0-7])++" + LONG_INTEGER_SUFFIX + "?+"))
      .withChannel(regexp(PythonTokenType.NUMBER, "0[xX](_?[0-9a-fA-F])++" + LONG_INTEGER_SUFFIX + "?+"))
      .withChannel(regexp(PythonTokenType.NUMBER, "0[bB](_?[01])++" + LONG_INTEGER_SUFFIX + "?+"))
      .withChannel(regexp(PythonTokenType.NUMBER, "[1-9](_?[0-9])*+" + LONG_INTEGER_SUFFIX + "?+"))
      .withChannel(regexp(PythonTokenType.NUMBER, "0(_?0)*+" + LONG_INTEGER_SUFFIX + "?+"))

      // http://docs.python.org/reference/lexical_analysis.html#identifiers
      .withChannel(new IdentifierAndKeywordChannel(and(
        or(IDENTIFIER_START, UNICODE_CHAR),
        o2n(or(IDENTIFIER_CONTINUE, UNICODE_CHAR))), true, PythonKeyword.values()))

      // http://docs.python.org/reference/lexical_analysis.html#operators
      // http://docs.python.org/reference/lexical_analysis.html#delimiters
      .withChannel(new PunctuatorChannel(PythonPunctuator.values()))

      .withChannel(new UnknownCharacterChannel());
  }
}
