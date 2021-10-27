/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
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
package org.sonar.python.regex;

import java.util.NoSuchElementException;
import javax.annotation.Nullable;
import org.sonarsource.analyzer.commons.regex.CharacterParser;
import org.sonarsource.analyzer.commons.regex.RegexSource;
import org.sonarsource.analyzer.commons.regex.ast.IndexRange;
import org.sonarsource.analyzer.commons.regex.ast.SourceCharacter;

public class PythonStringCharacterParser implements CharacterParser {

  final String sourceText;
  final int textLength;
  protected final RegexSource source;
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
      this.current = this.createCharAndUpdateIndex(this.sourceText.charAt(this.index), 1);
    }
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
}
