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

package org.sonar.plugins.python;

import com.fasterxml.jackson.core.JsonLocation;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;
import org.sonar.python.EscapeCharPositionInfo;

public class IpynbCellParser {

  public static final String SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER = "#SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER";
  private List<NotebookParsingData> accumulator = new ArrayList<>();
  private int accumulatedOffset = 0;

  public List<NotebookParsingData> getCellData() {
    return accumulator;
  }

  public void processCodeCell(JsonParser jParser) throws IOException {
    boolean isCodeCell = false;
    Optional<NotebookParsingData> notebookData = Optional.empty();
    while (jParser.nextToken() != JsonToken.END_OBJECT) {

      skipNestedObjects(jParser);

      if (!isCodeCell) {
        isCodeCell = processCodeCellType(jParser);
      }

      if (JsonToken.FIELD_NAME.equals(jParser.currentToken()) && "source".equals(jParser.currentName())) {
        jParser.nextToken();
        notebookData = Optional.of(parseCellData(jParser));
      }
    }

    if (isCodeCell && notebookData.isPresent()) {
      var data = notebookData.get();
      accumulator.add(data);
    }
  }

  private static void skipNestedObjects(JsonParser parser) throws IOException {
    if (parser.currentToken() == JsonToken.START_OBJECT || parser.currentToken() == JsonToken.START_ARRAY) {
      parser.skipChildren();
    }
  }

  private static boolean processCodeCellType(JsonParser jParser) throws IOException {
    if (JsonToken.FIELD_NAME.equals(jParser.currentToken()) && "cell_type".equals(jParser.currentName())) {
      jParser.nextToken();
      if ("code".equals(jParser.getValueAsString())) {
        return true;
      }
    }
    return false;
  }

  private NotebookParsingData parseCellData(JsonParser jParser) throws IOException {
    var startLine = computeStartLine(accumulator);
    var cellData = NotebookParsingData.fromLine(startLine);
    accumulatedOffset = 0;
    List<CellLine> rawLines;
    if (isJsonArrayCell(jParser)) {
      rawLines = extractLinesFromArray(jParser);
      addFromArrayToNotebookData(cellData, rawLines);
    } else {
      rawLines = extractLinesFromSingleString(jParser);
      addFromMultilineStringToNotebookData(cellData, rawLines);
    }

    JsonLocation lastLocation = jParser.currentTokenLocation();
    if (!rawLines.isEmpty()) {
      lastLocation = rawLines.get(rawLines.size() - 1).getTokenLocation();
    }
    cellData.addDelimiterToSource(SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER + "\n", lastLocation.getLineNr(), lastLocation.getColumnNr() + accumulatedOffset);
    return cellData;
  }

  private static int computeStartLine(List<NotebookParsingData> accumulator) {
    if (!accumulator.isEmpty()) {
      return accumulator.get(accumulator.size() - 1).getAggregatedSourceLine();
    }
    return 0;
  }

  private static boolean isJsonArrayCell(JsonParser jParser) {
    switch (jParser.currentToken()) {
      case START_ARRAY:
        return true;
      case VALUE_STRING:
        return false;
      default:
        throw new IllegalStateException("Unexpected token: " + jParser.currentToken());
    }
  }

  private static List<CellLine> extractLinesFromArray(JsonParser parser) throws IOException {
    List<CellLine> lines = new ArrayList<>();
    while (parser.nextToken() != JsonToken.END_ARRAY) {
      var line = parseCellLine(parser);
      lines.add(line);
    }
    // Only add an extra newline if the last element has an explicit \n
    if (!lines.isEmpty()) {
      var lastLine = lines.get(lines.size() - 1);
      if (lastLine.isLineEndingWithNewLine()) {
        lines.add(new CellLine("", lastLine.getTokenLocation(), List.of()));
      }
    }
    return lines;
  }

  private static CellLine parseCellLine(JsonParser parser) throws IOException {
    var content = parser.getValueAsString();
    var countEscapedChar = computeEscapeCharactersPositionInfo(content);
    return new CellLine(content, parser.currentTokenLocation(), countEscapedChar);
  }

  private static void addFromArrayToNotebookData(NotebookParsingData cellData, List<CellLine> rawLines) {
    var isCompressed = rawLines.stream().map(line -> line.getTokenLocation().getLineNr()).distinct().count() == 1;
    for (CellLine line : rawLines) {
      cellData.addLineToSource(line, 0, isCompressed);
    }
  }

  private static List<EscapeCharPositionInfo> computeEscapeCharactersPositionInfo(String sourceLine) {
    List<EscapeCharPositionInfo> escapeCharPositionInfoList = new LinkedList<>();
    var arr = sourceLine.toCharArray();
    for (int col = 0; col < sourceLine.length(); ++col) {
      char c = arr[col];
      if (c == '"' || c == '\\' || c == '\t' || c == '\b' || c == '\f') {
        escapeCharPositionInfoList.add(new EscapeCharPositionInfo(col, 1));
        // we never encounter \n or \r as the lines are split at these characters
      }
    }
    return escapeCharPositionInfoList;
  }

  private static List<CellLine> extractLinesFromSingleString(JsonParser parser) throws IOException {
    String content = parser.getValueAsString();
    boolean endsWithNewLine = content.endsWith("\n");
    // .lines() removes all the \n CellLine adds them back
    var lines = content.lines().map(line -> {
      var escapedCharInfo = computeEscapeCharactersPositionInfo(line);
      return new CellLine(line, parser.currentTokenLocation(), escapedCharInfo);
    }).collect(Collectors.toList());

    if (endsWithNewLine) {
      lines.add(new CellLine("", parser.currentTokenLocation(), List.of()));
    }
    return lines;
  }

  private void addFromMultilineStringToNotebookData(NotebookParsingData cellData, List<CellLine> rawLines) {
    for (CellLine line : rawLines) {
      cellData.addLineToSource(line, accumulatedOffset, true);
      var currentCount = line.getEscapedCharPositionInfo().stream().mapToInt(EscapeCharPositionInfo::numberOfExtraChars).sum();
      // as we add a \n it only counts a 1 char so we have to add one more
      var lineLength = line.getContent().length() + 1;
      accumulatedOffset += lineLength;
      accumulatedOffset += currentCount;
    }
  }

}
