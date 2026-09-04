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
package org.sonar.plugins.python;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.sonar.python.EscapeCharPositionInfo;
import org.sonar.python.IPythonLocation;

/**
 * Applies the leading-indentation cleanup performed by current IPython versions while keeping the
 * generated Python source aligned with the original notebook lines.
 *
 * <p>Since IPython 9.1, {@code leading_indent} delegates to Python's {@code textwrap.dedent}, which
 * removes the exact spaces/tabs prefix common to every non-blank line. Which lines count as blank
 * depends on the CPython version: since 3.14, {@code dedent} ignores every line containing only Python
 * whitespace; earlier versions ignored only space/tab-only lines, so other whitespace such as a CR
 * from a CRLF blank line participated in the margin. IPython 9.0 and earlier used the first line's
 * prefix after removing leading empty lines instead. A notebook records neither runtime version, so
 * this transformer follows the Python 3.14+ behavior while removing only a prefix shared by every
 * non-blank line.
 *
 * <p>IPython removes leading whitespace-only lines before {@code leading_indent}, and {@code dedent}
 * normalizes the remaining whitespace-only lines. This transformer instead preserves every such
 * physical line while excluding it from the common-margin calculation, keeping analyzer input and
 * notebook source mappings stable without affecting Python's block structure.
 */
final class IPythonCellInputTransformer {

  private IPythonCellInputTransformer() {
  }

  static NotebookParsingData removeCommonLeadingIndentation(NotebookParsingData cellData) {
    String source = cellData.getAggregatedSource().toString();
    String margin = commonLeadingIndentation(source);
    if (margin.isEmpty()) {
      return cellData;
    }

    StringBuilder transformedSource = new StringBuilder(source.length());
    Map<Integer, IPythonLocation> transformedLocations = new LinkedHashMap<>(cellData.getLocationMap());
    int generatedLine = transformedLocations.keySet().stream().mapToInt(Integer::intValue).min().orElse(0);

    int lineStart = 0;
    while (lineStart < source.length()) {
      int lineEnd = findLineEnd(source, lineStart);
      boolean containsOnlyWhitespace = containsOnlyPythonWhitespace(source, lineStart, lineEnd);
      boolean startsWithMargin = source.startsWith(margin, lineStart);
      int removedCharacters = containsOnlyWhitespace || !startsWithMargin ? 0 : margin.length();

      transformedSource.append(source, lineStart + removedCharacters, lineEnd);
      if (lineEnd < source.length()) {
        transformedSource.append('\n');
      }
      if (generatedLine > 0) {
        shiftLocation(transformedLocations, generatedLine, removedCharacters);
        generatedLine++;
      }
      lineStart = lineEnd + 1;
    }

    return new NotebookParsingData(transformedSource, transformedLocations, cellData.getAggregatedSourceLine());
  }

  private static String commonLeadingIndentation(String source) {
    String margin = null;
    int lineStart = 0;
    while (lineStart < source.length()) {
      int lineEnd = findLineEnd(source, lineStart);
      if (!containsOnlyPythonWhitespace(source, lineStart, lineEnd)) {
        int indentationEnd = leadingIndentationEnd(source, lineStart, lineEnd);
        String indentation = source.substring(lineStart, indentationEnd);
        margin = margin == null ? indentation : commonPrefix(margin, indentation);
        if (margin.isEmpty()) {
          return "";
        }
      }
      lineStart = lineEnd + 1;
    }
    return margin == null ? "" : margin;
  }

  private static int findLineEnd(String source, int lineStart) {
    int newline = source.indexOf('\n', lineStart);
    return newline >= 0 ? newline : source.length();
  }

  private static int leadingIndentationEnd(String source, int start, int end) {
    int index = start;
    while (index < end && (source.charAt(index) == ' ' || source.charAt(index) == '\t')) {
      index++;
    }
    return index;
  }

  private static boolean containsOnlyPythonWhitespace(String source, int start, int end) {
    for (int index = start; index < end; index++) {
      char character = source.charAt(index);
      // Java's two whitespace predicates together cover Python's str.isspace() characters except NEL.
      if (!Character.isWhitespace(character) && !Character.isSpaceChar(character) && character != '\u0085') {
        return false;
      }
    }
    return true;
  }

  private static String commonPrefix(String left, String right) {
    int length = Math.min(left.length(), right.length());
    int index = 0;
    while (index < length && left.charAt(index) == right.charAt(index)) {
      index++;
    }
    return left.substring(0, index);
  }

  private static void shiftLocation(Map<Integer, IPythonLocation> locations, int generatedLine, int removedCharacters) {
    IPythonLocation location = locations.get(generatedLine);
    if (location == null || removedCharacters == 0) {
      return;
    }
    int removedEscapeCharacters = location.colOffsets().stream()
      .filter(offset -> offset.columnInIpynbFile() < removedCharacters)
      .mapToInt(EscapeCharPositionInfo::numberOfExtraChars)
      .sum();
    List<EscapeCharPositionInfo> shiftedOffsets = location.colOffsets().stream()
      .filter(offset -> offset.columnInIpynbFile() >= removedCharacters)
      .map(offset -> new EscapeCharPositionInfo(offset.columnInIpynbFile() - removedCharacters, offset.numberOfExtraChars()))
      .toList();
    locations.put(generatedLine, new IPythonLocation(
      location.line(),
      location.column() + removedCharacters + removedEscapeCharacters,
      shiftedOffsets,
      location.isCompresssed()));
  }
}
