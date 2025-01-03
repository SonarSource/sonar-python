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
package org.sonar.python.regex;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.plugins.python.api.tree.StringElement;
import org.sonar.plugins.python.api.tree.Token;
import org.sonarsource.analyzer.commons.regex.CharacterParser;
import org.sonarsource.analyzer.commons.regex.ast.IndexRange;
import org.sonarsource.analyzer.commons.regex.python.PythonRegexSource;

public class PythonAnalyzerRegexSource extends PythonRegexSource {

  private static final IndexRange OPENER_RANGE = new IndexRange(-1, 0);
  private final int sourceLine;
  private final int sourceStartOffset;
  private final int[] lineStartOffsets;

  private final boolean isRawString;

  public PythonAnalyzerRegexSource(StringElement s) {
    super(s.trimmedQuotesValue());
    String prefix = s.prefix();
    Token firstToken = s.firstToken();
    sourceLine = firstToken.line();
    sourceStartOffset = firstToken.column() + (s.isTripleQuoted() ? 3 : 1) + prefix.length();
    lineStartOffsets = lineStartOffsets(getSourceText());
    isRawString = prefix.toLowerCase(Locale.ROOT).contains("r");
  }

  @Override
  public CharacterParser createCharacterParser() {
    return new PythonStringCharacterParser(this);
  }

  public LocationInFile locationInFileFor(IndexRange range) {
    if (OPENER_RANGE.equals(range)) {
      return new LocationInFile(null, sourceLine, sourceStartOffset - 2, sourceLine, sourceStartOffset - 1);
    }

    int[] startLineAndOffset = lineAndOffset(range.getBeginningOffset());
    int[] endLineAndOffset = lineAndOffset(range.getEndingOffset());
    return new LocationInFile(null, startLineAndOffset[0], startLineAndOffset[1], endLineAndOffset[0], endLineAndOffset[1]);
  }

  public boolean isRawString() {
    return isRawString;
  }

  private int[] lineAndOffset(int index) {
    int line;
    int offset;
    int searchResult = Arrays.binarySearch(lineStartOffsets, index);
    if (searchResult >= 0) {
      line = sourceLine + searchResult;
      offset = 0;
    } else {
      line = sourceLine - searchResult - 2;
      offset = index - lineStartOffsets[- searchResult - 2];
    }
    if (line == sourceLine) {
      offset += sourceStartOffset;
    }
    return new int[] { line, offset };
  }

  private static int[] lineStartOffsets(String text) {
    List<Integer> lineStartOffsets = new ArrayList<>();
    lineStartOffsets.add(0);
    int length = text.length();
    int i = 0;
    while (i < length) {
      if (text.charAt(i) == '\n' || text.charAt(i) == '\r') {
        int nextLineStartOffset = i + 1;
        if (i < (length - 1) && text.charAt(i) == '\r' && text.charAt(i + 1) == '\n') {
          nextLineStartOffset = i + 2;
          i++;
        }
        lineStartOffsets.add(nextLineStartOffset);
      }
      i++;
    }
    return lineStartOffsets.stream().mapToInt(x -> x).toArray();
  }
}
