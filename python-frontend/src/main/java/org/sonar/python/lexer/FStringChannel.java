/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
import com.sonar.sslr.api.TokenType;
import com.sonar.sslr.impl.Lexer;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.stream.IntStream;
import org.sonar.python.api.PythonPunctuator;
import org.sonar.python.api.PythonTokenType;
import org.sonar.python.lexer.FStringState.Mode;
import org.sonar.sslr.channel.Channel;
import org.sonar.sslr.channel.CodeReader;

/**
 * A channel to handle f-strings.
 * See https://docs.python.org/3.12/reference/lexical_analysis.html#formatted-string-literals
 */
public class FStringChannel extends Channel<Lexer> {

  private static final char EOF = (char) -1;

  private final LexerState lexerState;
  private final StringBuilder sb = new StringBuilder();

  private static final Set<Character> QUOTES = Set.of('\"', '\'');
  private static final Set<Character> PREFIXES = Set.of('F', 'R');
  private static final Set<String> ESCAPED_CHARS = Set.of("{{", "}}");

  public FStringChannel(LexerState lexerState) {
    this.lexerState = lexerState;
  }

  @Override
  public boolean consume(CodeReader code, Lexer output) {
    char c = code.charAt(0);
    int line = code.getLinePosition();
    int column = code.getColumnPosition();

    FStringState currentState = lexerState.fStringStateStack.peek();

    if (canConsumeFStringPrefix(sb, code)) {
      char quote = code.charAt(0);
      StringBuilder quotes = consumeFStringQuotes(code, quote);
      boolean isRawString = sb.indexOf("r") >= 0 || sb.indexOf("R") >= 0;
      FStringState newState = new FStringState(Mode.FSTRING_MODE, lexerState.brackets, isRawString);
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
      // because the lexerState removes one to the count of brackets before entering this channel
      // we need to adjust the comparison
      if (c == '}' && currentState.getBrackets() - 1 == lexerState.brackets) {
        Token rCurlyBraceToken = buildToken(PythonPunctuator.RCURLYBRACE, "}", output, line, column);
        code.pop();
        List<Token> tokens = new ArrayList<>();
        tokens.add(rCurlyBraceToken);
        lexerState.fStringStateStack.pop();
        FStringState previousState = lexerState.fStringStateStack.peek();
        return consumeFStringMiddle(tokens, sb, previousState, code, output);
        // do not lex colon if the nesting level is different from the open curly brace
      } else if (c == ':' && lexerState.brackets == currentState.getBrackets()) {
        Token formatSpecifier = buildToken(PythonPunctuator.COLON, ":", output, line, column);
        code.pop();
        List<Token> tokens = new ArrayList<>();
        tokens.add(formatSpecifier);
        FStringState newState = new FStringState(Mode.FORMAT_SPECIFIER_MODE, lexerState.brackets, currentState.isRawString);
        lexerState.fStringStateStack.push(newState);
        return consumeFStringMiddle(tokens, sb, newState, code, output);
      }
    }
    return false;
  }

  private boolean consumeFStringMiddle(List<Token> tokens, StringBuilder sb, FStringState state, CodeReader code, Lexer output) {
    int line = code.getLinePosition();
    int column = code.getColumnPosition();
    FStringState.Mode currentMode = state.getTokenizerMode();
    while (code.charAt(0) != EOF) {
      // In a raw string we consider \ as a character not as escape so we consume it as is.
      // Except for quotes which will be consumed as an escaped char
      if (currentMode == Mode.FSTRING_MODE && isRawStringSingleBackSlash(code, state)) {
        sb.append((char) code.pop());
        // If we encounter an escaped char we can consume the next two chars directly
        // Or if we encounter two \\
      } else if (currentMode == Mode.FSTRING_MODE && (isEscapedChar(code) || isDoubleBackslashInRawString(state, code))) {
        sb.append((char) code.pop());
        sb.append((char) code.pop());
      } else if (code.charAt(0) == '{' && !isUnicodeChar(sb)) {
        addFStringMiddleToTokens(tokens, sb, output, line, column);
        addLCurlBraceAndSwitchToRegularMode(tokens, code, output, state);
        addTokens(tokens, output);
        return true;
      } else if (currentMode == Mode.FORMAT_SPECIFIER_MODE && code.charAt(0) == '}') {
        addFStringMiddleToTokens(tokens, sb, output, line, column);
        lexerState.fStringStateStack.pop();
        addTokens(tokens, output);
        return true;
      } else if (currentMode == Mode.FSTRING_MODE && areClosingQuotes(code, state)) {
        addFStringMiddleToTokens(tokens, sb, output, line, column);
        addFStringEndToTokens(code, state.getQuote(), tokens, output);
        addTokens(tokens, output);
        return true;
      } else {
        sb.append((char) code.pop());
      }
    }
    return false;
  }

  private static boolean isDoubleBackslashInRawString(FStringState state, CodeReader code) {
    return state.isRawString && code.charAt(0) == '\\' && code.charAt(1) == '\\';
  }

  private static boolean canConsumeFStringPrefix(StringBuilder sb, CodeReader code) {
    Character firstChar = Character.toUpperCase(code.charAt(0));
    Character secondChar = Character.toUpperCase(code.charAt(1));
    if (firstChar == 'F' && QUOTES.contains(code.charAt(1))) {
      sb.append((char) code.pop());
      return true;
    } else if (PREFIXES.contains(firstChar) && PREFIXES.contains(secondChar) &&
      !firstChar.equals(secondChar) && QUOTES.contains(code.charAt(2))) {
        sb.append((char) code.pop());
        sb.append((char) code.pop());
        return true;
      }
    return false;
  }

  private static boolean isRawStringSingleBackSlash(CodeReader code, FStringState state) {
    return state.isRawString && code.charAt(0) == '\\' && !QUOTES.contains(code.charAt(1)) && code.charAt(1) != '\\';
  }

  private static boolean isUnicodeChar(StringBuilder sb) {
    int lastIndexOfUnicodeChar = sb.lastIndexOf("\\N");
    return lastIndexOfUnicodeChar >= 0 && lastIndexOfUnicodeChar == sb.length() - 2;
  }

  private static boolean isEscapedChar(CodeReader code) {
    return ESCAPED_CHARS.contains(String.valueOf(code.peek(2))) || code.peek() == '\\';
  }

  private static boolean areClosingQuotes(CodeReader code, FStringState state) {
    char[] quotes = code.peek(state.getNumberOfQuotes());
    return IntStream.range(0, quotes.length).mapToObj(i -> quotes[i]).allMatch(state.getQuote()::equals);
  }

  private static void addFStringMiddleToTokens(List<Token> tokens, StringBuilder sb, Lexer output, int line, int column) {
    if (sb.length() != 0) {
      Token fStringMiddleToken = buildToken(PythonTokenType.FSTRING_MIDDLE, sb.toString(), output, line, column);
      sb.setLength(0);
      tokens.add(fStringMiddleToken);
    }
  }

  private void addFStringEndToTokens(CodeReader code, char quote, List<Token> tokens, Lexer output) {
    int line = code.getLinePosition();
    int column = code.getColumnPosition();
    StringBuilder endQuotes = consumeFStringQuotes(code, quote);
    lexerState.fStringStateStack.pop();
    Token fStringEndToken = buildToken(PythonTokenType.FSTRING_END, endQuotes.toString(), output, line, column);
    tokens.add(fStringEndToken);
  }

  private void addLCurlBraceAndSwitchToRegularMode(List<Token> tokens, CodeReader code, Lexer output, FStringState currentState) {
    Token curlyBraceToken = buildToken(PythonPunctuator.LCURLYBRACE, "{", output, code.getLinePosition(), code.getColumnPosition());
    code.pop();
    lexerState.brackets++;
    FStringState updatedState = new FStringState(FStringState.Mode.REGULAR_MODE, lexerState.brackets, currentState.isRawString);
    lexerState.fStringStateStack.push(updatedState);
    tokens.add(curlyBraceToken);
  }

  private static StringBuilder consumeFStringQuotes(CodeReader code, char quote) {
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

  private static void addTokens(List<Token> tokens, Lexer output) {
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
}
