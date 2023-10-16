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
package org.sonar.python.lexer;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

import org.sonar.python.api.PythonPunctuator;
import org.sonar.python.api.PythonTokenType;
import org.sonar.python.lexer.FStringState.Mode;
import org.sonar.sslr.channel.Channel;
import org.sonar.sslr.channel.CodeReader;

import com.sonar.sslr.api.Token;
import com.sonar.sslr.api.TokenType;
import com.sonar.sslr.impl.Lexer;

/**
 * A channel to handle f-strings.
 * See https://docs.python.org/3.12/reference/lexical_analysis.html#formatted-string-literals
 */
public class FStringChannel312 extends Channel<Lexer> {

  private static final char EOF = (char) -1;

  private final LexerState lexerState;
  private final StringBuilder sb = new StringBuilder();

  private static final List<Character> QUOTES = List.of('\"', '\'');
  private static final List<Character> PREFIXES = List.of('F', 'R');

  public FStringChannel312(LexerState lexerState) {
    this.lexerState = lexerState;
  }

  @Override
  public boolean consume(CodeReader code, Lexer output) {
    setInitialLineAndColumn(code);

    char c = code.charAt(0);
    int line = code.getLinePosition();
    int column = code.getColumnPosition();

    FStringState currentState = lexerState.fStringStateStack.peek();

    if (canConsumeFStringPrefix(sb, code)) {
      char quote = (char) code.charAt(0);
      StringBuilder quotes = consumeFStringQuotes(code, quote);
      FStringState newState = new FStringState(Mode.FSTRING_MODE);
      newState.setQuote(quote);
      newState.setNumberOfQuotes(quotes.length());
      lexerState.fStringStateStack.push(newState);
      Token fStringStartToken = buildToken(PythonTokenType.FSTRING_START, sb.append(quotes).toString(), output, line, column);
      sb.setLength(0);
      List<Token> tokens = new ArrayList<>();
      tokens.add(fStringStartToken);
      return consumeFStringMiddle(tokens, sb, newState, code, output);
    }

    FStringState.Mode currentMode = currentState.getTokenizerMode();

    if (currentMode == Mode.REGULAR_MODE && lexerState.fStringStateStack.size() > 1) {
      if (c == '}') {
        Token rCurlyBraceToken = buildToken(PythonPunctuator.RCURLYBRACE, "}", output, line, column);
        code.pop();
        List<Token> tokens = new ArrayList<>();
        tokens.add(rCurlyBraceToken);
        lexerState.fStringStateStack.pop();
        FStringState previousState = lexerState.fStringStateStack.peek();
        return consumeFStringMiddle(tokens, sb, previousState, code, output);
      } else if (c == ':') {
        Token formatSpecifier = buildToken(PythonPunctuator.COLON, ":", output, line, column);
        code.pop();
        List<Token> tokens = new ArrayList<>();
        tokens.add(formatSpecifier);
        FStringState newState = new FStringState(Mode.FORMAT_SPECIFIER_MODE);
        lexerState.fStringStateStack.push(newState);
        return consumeFStringMiddle(tokens, sb, newState, code, output);
      }
    } else {
      if (c == currentState.getQuote()) {
        StringBuilder quotes = consumeFStringQuotes(code, currentState.getQuote());
        lexerState.fStringStateStack.pop();
        addToken(PythonTokenType.FSTRING_END, quotes.toString(), output, line, column);
        return true;
      }
    }
    return false;
  }


  private boolean consumeFStringMiddle(List<Token> tokens, StringBuilder sb, FStringState state, CodeReader code, Lexer output) {
    int line = code.getLinePosition();
    int column = code.getColumnPosition();
    FStringState.Mode currentMode = state.getTokenizerMode();
    while (code.charAt(0) != EOF) {
      if (currentMode == Mode.FSTRING_MODE && isEscapedCurlyBrace(code)) {
        code.pop();
        sb.append((char) code.pop());
      } else if (code.charAt(0) == '{') {
        addFStringMiddleToTokens(tokens, sb, output, line, column);
        addLCurlBraceAndSwitchToRegularMode(tokens, code, output);
        addTokens(tokens, output);
        return true;
      } else if (currentMode == Mode.FORMAT_SPECIFIER_MODE && code.charAt(0) == '}') {
        addFStringMiddleToTokens(tokens, sb, output, line, column);
        lexerState.fStringStateStack.pop();
        addTokens(tokens, output);
        return true;
      } else if (currentMode == Mode.FSTRING_MODE && areClosingQuotes(code, state)) {
        addFStringMiddleToTokens(tokens, sb, output, line, column);
        addFStringEndToTokens(code, state.getQuote(), tokens, sb, output);
        addTokens(tokens, output);
        return true;
      } else {
        sb.append((char) code.pop());
      }
    }
    return false;
  }

  private boolean canConsumeFStringPrefix(StringBuilder sb, CodeReader code) {
    if (PREFIXES.contains(Character.toUpperCase(code.charAt(0)))) {
      if (QUOTES.contains(code.charAt(1))) {
        sb.append((char) code.pop());
        return true;
      } else if ((PREFIXES.contains(Character.toUpperCase(code.charAt(1)))) && QUOTES.contains(code.charAt(2))) {
        sb.append((char) code.pop());
        sb.append((char) code.pop());
        return true;
      }
    }
    return false;
  }

  private boolean isEscapedCurlyBrace(CodeReader code) {
    return (code.charAt(0) == '{' && code.charAt(1) == '{') || (code.charAt(0) == '}' && code.charAt(1) == '}');
  }

  private boolean areClosingQuotes(CodeReader code, FStringState state) {
    char[] quotes = code.peek(state.getNumberOfQuotes());
    return IntStream.range(0, quotes.length).mapToObj(i -> quotes[i]).allMatch(c -> c == state.getQuote());
  }

  private void addFStringMiddleToTokens(List<Token> tokens, StringBuilder sb, Lexer output, int line, int column) {
    if (sb.length() != 0) {
      Token fStringMiddleToken = buildToken(PythonTokenType.FSTRING_MIDDLE, sb.toString(), output, line, column);
      sb.setLength(0);
      tokens.add(fStringMiddleToken);
    }
  }

  private void addFStringEndToTokens(CodeReader code, char quote, List<Token> tokens, StringBuilder sb, Lexer output) {
    StringBuilder endQuotes = consumeFStringQuotes(code, quote);
    lexerState.fStringStateStack.pop();
    Token fStringEndToken = buildToken(PythonTokenType.FSTRING_END, endQuotes.toString(), output, code.getLinePosition(), code.getColumnPosition());
    tokens.add(fStringEndToken);
  }

  private void addLCurlBraceAndSwitchToRegularMode(List<Token> tokens, CodeReader code, Lexer output) {
    Token curlyBraceToken = buildToken(PythonPunctuator.LCURLYBRACE, "{", output, code.getLinePosition(), code.getColumnPosition());
    code.pop();
    FStringState updatedState = new FStringState(FStringState.Mode.REGULAR_MODE);
    lexerState.fStringStateStack.push(updatedState);
    tokens.add(curlyBraceToken);
  }

  private StringBuilder consumeFStringQuotes(CodeReader code, char quote) {
    StringBuilder quotes = new StringBuilder();
    if (code.charAt(1) == quote && code.charAt(2) == quote) {
      quotes.append((char) code.pop());
      quotes.append((char) code.pop());
      quotes.append((char) code.pop());
    } else {
      quotes.append((char) code.pop());
    }
    return quotes;
  }

  private static void addToken(TokenType tokenType, String value, Lexer output, int line, int column) {
    output.addToken(buildToken(tokenType, value, output, line, column));
  }

  private void addTokens(List<Token> tokens, Lexer output) {
    output.addToken(tokens.toArray(Token[]::new));
  }

  private static Token buildToken(TokenType tokenType, String value, Lexer output, int line, int column) {
    return Token.builder()
      .setType(tokenType)
      .setValueAndOriginalValue(value)
      .setURI(output.getURI())
      .setLine(line)
      .setColumn(column)
      .build();
  }

  private void setInitialLineAndColumn(CodeReader code) {
    if (code.getLinePosition() == 1 && code.getColumnPosition() == 0) {
      code.setLinePosition(lexerState.initialLine);
      code.setColumnPosition(lexerState.initialColumn);
    }
  }
}
