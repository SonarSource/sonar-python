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

public class FStringState {

  Character quote;
  int numberOfQuotes;
  int brackets;
  boolean isRawString;


  public enum Mode {
    REGULAR_MODE,
    FSTRING_MODE,
    FORMAT_SPECIFIER_MODE
  }

  private Mode tokenizerMode;

  public FStringState(Mode mode, int brackets, boolean isRawString) {
    this.tokenizerMode = mode;
    this.brackets = brackets;
    this.isRawString = isRawString;
  }

  public Character getQuote() {
    return quote;
  }

  public void setQuote(Character quote) {
    this.quote = quote;
  }

  public Mode getTokenizerMode() {
    return tokenizerMode;
  }

  public int getNumberOfQuotes() {
    return numberOfQuotes;
  }

  public void setNumberOfQuotes(int numberOfQuotes) {
    this.numberOfQuotes = numberOfQuotes;
  }

  public int getBrackets() {
    return brackets;
  }
}
