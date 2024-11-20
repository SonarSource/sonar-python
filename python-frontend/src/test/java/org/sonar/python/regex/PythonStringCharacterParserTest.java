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

import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.tree.StringElement;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.python.PythonTestUtils;
import org.sonarsource.analyzer.commons.regex.CharacterParser;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class PythonStringCharacterParserTest {

  @Test
  void test_first_character_on_init() {
    assertThat(getCharacterParser("'abcde'").getCurrent().getCharacter()).isEqualTo('a');
  }

  @Test
  void test_move_next() {
    CharacterParser characterParser = getCharacterParser("'abcde'");
    characterParser.moveNext();
    assertThat(characterParser.getCurrent().getCharacter()).isEqualTo('b');
  }

  @Test
  void test_move_next_to_the_end() {
    CharacterParser characterParser = getCharacterParser("'ab'");
    characterParser.moveNext();
    assertThat(characterParser.isAtEnd()).isFalse();
    characterParser.moveNext();
    assertThat(characterParser.isAtEnd()).isTrue();
    assertThatThrownBy(characterParser::getCurrent).isInstanceOf(NoSuchElementException.class);
  }

  @Test
  void test_reset_to() {
    CharacterParser characterParser = getCharacterParser("'abcde'");
    characterParser.moveNext();
    characterParser.resetTo(0);
    assertThat(characterParser.getCurrent().getCharacter()).isEqualTo('a');
  }

  @Test
  void test_escaping_has_no_meaning_in_raw_string() {
    assertThat(chars("r'\\n'")).containsExactly('\\', 'n');
  }

  @Test
  void test_different_escape_sequences() {
    assertThat(chars("'a\\\nb'")).containsExactly('a', 'b');
    assertThat(chars("'a\\\n'")).containsExactly('a');
    assertThat(chars("'\\\\'")).containsExactly('\\');
    assertThat(chars("'\\''")).containsExactly('\'');
    assertThat(chars("'\\\"'")).containsExactly('"');
    assertThat(chars("'\\a'")).containsExactly('\u0007');
    assertThat(chars("'\\b'")).containsExactly('\b');
    assertThat(chars("'\\f'")).containsExactly('\f');
    assertThat(chars("'\\n'")).containsExactly('\n');
    assertThat(chars("'\\r'")).containsExactly('\r');
    assertThat(chars("'\\t'")).containsExactly('\t');
    assertThat(chars("'\\v'")).containsExactly('\u000b');
    assertThat(chars("'\\u0041'")).containsExactly('A');
    assertThat(chars("'\\U00000041'")).containsExactly('A');
    assertThat(chars("'\\x41'")).containsExactly('A');
    assertThat(chars("'\\101'")).containsExactly('A');

    assertThat(chars("'\\y'")).containsExactly('\\', 'y');
  }

  @Test
  void invalid_escape_sequences() {
    assertThat(chars("'\\x4'")).containsExactly('\\', 'x', '4');
    assertThat(chars("'\\u4'")).containsExactly('\\', 'u', '4');
    assertThat(chars("'\\U4'")).containsExactly('\\', 'U', '4');
  }

  private List<Character> chars(String s) {
    CharacterParser characterParser = getCharacterParser(s);
    List<Character> result = new ArrayList<>();
    while (characterParser.isNotAtEnd()) {
      result.add(characterParser.getCurrent().getCharacter());
      characterParser.moveNext();
    }
    return result;
  }

  private CharacterParser getCharacterParser(String s) {
    PythonAnalyzerRegexSource regexSource = new PythonAnalyzerRegexSource(stringElement(s));
    return regexSource.createCharacterParser();
  }

  private static StringElement stringElement(String code) {
    return ((StringLiteral) PythonTestUtils.lastExpression(code)).stringElements().get(0);
  }
}
