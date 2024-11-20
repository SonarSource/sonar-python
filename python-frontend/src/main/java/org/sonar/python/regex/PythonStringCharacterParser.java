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
package org.sonar.python.regex;

import java.util.NoSuchElementException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import javax.annotation.Nullable;
import org.sonarsource.analyzer.commons.regex.CharacterParser;
import org.sonarsource.analyzer.commons.regex.ast.IndexRange;
import org.sonarsource.analyzer.commons.regex.ast.SourceCharacter;

public class PythonStringCharacterParser implements CharacterParser {

  private static final Pattern UNICODE_16_BIT_PATTERN = Pattern.compile("\\Au([0-9A-Fa-f]{4})");
  private static final Pattern UNICODE_32_BIT_PATTERN = Pattern.compile("\\AU([0-9A-Fa-f]{8})");
  private static final Pattern HEX_PATTERN = Pattern.compile("\\Ax([0-9A-Fa-f]{2})");
  private static final Pattern OCTAL_PATTERN = Pattern.compile("\\A([0-7]{1,3})");

  final String sourceText;
  final int textLength;
  protected final PythonAnalyzerRegexSource source;
  protected int index;
  @Nullable
  private SourceCharacter current;

  public PythonStringCharacterParser(PythonAnalyzerRegexSource source) {
    this.source = source;
    sourceText = source.getSourceText();
    textLength = source.length();
    index = 0;
    this.moveNext();
  }

  @Override
  public void moveNext() {
    if (this.index >= this.textLength) {
      this.current = null;
    } else {
      this.current = parsePythonCharacter();
    }
  }

  private SourceCharacter parsePythonCharacter() {
    char ch = sourceText.charAt(index);
    if (!source.isRawString() && ch == '\\') {
      if (index + 1 >= textLength) {
        return createCharAndUpdateIndex('\\', 1);
      }
      return parsePythonEscapeSequence();
    }
    return createCharAndUpdateIndex(ch, 1);
  }

  SourceCharacter createCharAndUpdateIndex(char ch, int length) {
    int startIndex = this.index;
    this.index += length;
    return new SourceCharacter(this.source, new IndexRange(startIndex, this.index), ch, length > 1);
  }

  @Override
  public SourceCharacter getCurrent() {
    if (this.current == null) {
      throw new NoSuchElementException();
    } else {
      return this.current;
    }
  }

  @Override
  public boolean isAtEnd() {
    return this.current == null;
  }

  @Override
  public void resetTo(int index) {
    this.index = index;
    this.moveNext();
  }

  private SourceCharacter parsePythonEscapeSequence() {
    char charAfterBackslash = sourceText.charAt(index + 1);
    switch (charAfterBackslash) {
      case '\n':
        // \NEWLINE is ignored in python. We skip both characters
        if (this.index + 2 >= this.textLength) {
          return null;
        }
        this.index += 2;
        this.moveNext();
        return getCurrent();
      case '\\':
        return createCharAndUpdateIndex('\\', 2);
      case '\'':
        return createCharAndUpdateIndex('\'', 2);
      case '"':
        return createCharAndUpdateIndex('"', 2);
      case 'a':
        return createCharAndUpdateIndex('\u0007', 2);
      case 'b':
        return createCharAndUpdateIndex('\b', 2);
      case 'f':
        return createCharAndUpdateIndex('\f', 2);
      case 'n':
        return createCharAndUpdateIndex('\n', 2);
      case 'r':
        return createCharAndUpdateIndex('\r', 2);
      case 't':
        return createCharAndUpdateIndex('\t', 2);
      case 'v':
        return createCharAndUpdateIndex('\u000b', 2);
      case 'u':
        return createCharacterFromPattern(UNICODE_16_BIT_PATTERN, 16, 2);
      case 'U':
        return createCharacterFromPattern(UNICODE_32_BIT_PATTERN, 16, 2);
      case 'x':
        return createCharacterFromPattern(HEX_PATTERN, 16, 2);
      default:
        return createCharacterFromPattern(OCTAL_PATTERN, 8, 1);
    }
  }

  private SourceCharacter createCharacterFromPattern(Pattern pattern, int radix, int initialLength) {
    Matcher matcher = pattern.matcher(sourceText.substring(index + 1));
    if (matcher.find()) {
      String value = matcher.group(1);
      return createCharAndUpdateIndex((char) Integer.parseInt(value, radix), value.length() + initialLength);
    }
    return createCharAndUpdateIndex('\\', 1);
  }
}
