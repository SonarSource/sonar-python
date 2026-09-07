/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.python.parser;

import java.util.Comparator;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

public record IPythonParserConfiguration(List<OpaqueCellRange> opaqueCellRanges) {

  private static final IPythonParserConfiguration EMPTY = new IPythonParserConfiguration(List.of());

  public IPythonParserConfiguration {
    opaqueCellRanges = opaqueCellRanges.stream()
      .sorted(Comparator.comparingInt(OpaqueCellRange::startLine))
      .toList();
    for (int i = 1; i < opaqueCellRanges.size(); i++) {
      if (opaqueCellRanges.get(i - 1).delimiterLine() >= opaqueCellRanges.get(i).startLine()) {
        throw new IllegalArgumentException("Opaque cell ranges must be separated by a delimiter line");
      }
    }
  }

  /**
   * Retained as a convenience for callers that only need to identify which cells are opaque.
   */
  public Set<Integer> cellMagicStartLines() {
    return opaqueCellRanges.stream()
      .map(OpaqueCellRange::startLine)
      .collect(Collectors.toUnmodifiableSet());
  }

  public static IPythonParserConfiguration empty() {
    return EMPTY;
  }

  /**
   * A generated-source range whose content must not be interpreted by the Python lexer.
   * The delimiter line is excluded from the range so that it remains a structural token.
   */
  public record OpaqueCellRange(int startLine, int delimiterLine) {

    public OpaqueCellRange {
      if (startLine < 1 || delimiterLine <= startLine) {
        throw new IllegalArgumentException("An opaque cell range must end after a positive start line");
      }
    }
  }
}
