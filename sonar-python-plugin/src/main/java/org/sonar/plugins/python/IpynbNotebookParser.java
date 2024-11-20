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
package org.sonar.plugins.python;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonLocation;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import org.sonar.python.EscapeCharPositionInfo;
import org.sonar.python.IPythonLocation;

public class IpynbNotebookParser {

  public static final String SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER = "#SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER";

  private static final Set<String> ACCEPTED_LANGUAGE = Set.of("python", "ipython");


  public static Optional<GeneratedIPythonFile> parseNotebook(PythonInputFile inputFile) {
    try {
      return new IpynbNotebookParser(inputFile).parse();
    } catch (IOException e) {
      throw new IllegalStateException("Cannot read " + inputFile, e);
    }
  }

  private IpynbNotebookParser(PythonInputFile inputFile) {
    this.inputFile = inputFile;
  }

  private final PythonInputFile inputFile;

  private int lastPythonLine = 0;

  public Optional<GeneratedIPythonFile> parse() throws IOException {
    // If the language is not present, we assume it is a Python notebook
    var isPythonNotebook = parseLanguage().map(ACCEPTED_LANGUAGE::contains).orElse(true);

    return Boolean.TRUE.equals(isPythonNotebook) ? Optional.of(parseNotebook()) : Optional.empty();
  }

  public Optional<String> parseLanguage() throws IOException {
    String content = inputFile.wrappedFile().contents();
    JsonFactory factory = new JsonFactory();
    try (JsonParser jParser = factory.createParser(content)) {
      while (!jParser.isClosed()) {
        JsonToken jsonToken = jParser.nextToken();
        if (JsonToken.FIELD_NAME.equals(jsonToken) && "language".equals(jParser.currentName())) {
          jParser.nextToken();
          return Optional.ofNullable(jParser.getValueAsString());
        }
      }
    }
    return Optional.empty();
  }

  public GeneratedIPythonFile parseNotebook() throws IOException {
    String content = inputFile.wrappedFile().contents();
    boolean isCompressed = content.lines().count() <= 1;
    JsonFactory factory = new JsonFactory();
    try (JsonParser jParser = factory.createParser(content)) {
      return parseCells(jParser, isCompressed).map(notebookData -> {
        // Account for EOF token
        JsonLocation location = jParser.currentTokenLocation();
        notebookData.addDefaultLocation(lastPythonLine, location.getLineNr(), location.getColumnNr());
        return new GeneratedIPythonFile(inputFile.wrappedFile(), notebookData.getAggregatedSource().toString(), notebookData.getLocationMap());
      }).orElse(new GeneratedIPythonFile(inputFile.wrappedFile(), "", new LinkedHashMap<>()));
    }

  }

  private Optional<NotebookParsingData> parseCells(JsonParser parser, boolean isCompressed) throws IOException {
    while (!parser.isClosed()) {
      parser.nextToken();
      String fieldName = parser.currentName();
      if ("cells".equals(fieldName)) {
        // consume array start token
        parser.nextToken();
        Optional<NotebookParsingData> data = parseCellArray(parser, isCompressed);
        parser.close();
        return data;
      }
    }
    return Optional.empty();
  }

  private Optional<NotebookParsingData> parseCellArray(JsonParser jParser, boolean isCompressed) throws IOException {
    List<NotebookParsingData> cellsData = new ArrayList<>();

    while (jParser.nextToken() != JsonToken.END_ARRAY) {
      if (jParser.currentToken() == JsonToken.START_OBJECT) {
        processCodeCell(cellsData, jParser, isCompressed);
      }
    }
    Optional<NotebookParsingData> aggregatedNotebookData = cellsData.stream().reduce(NotebookParsingData::combine);
    aggregatedNotebookData.ifPresent(NotebookParsingData::removeTrailingExtraLine);
    return aggregatedNotebookData;
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

  private void processCodeCell(List<NotebookParsingData> accumulator, JsonParser jParser, boolean isCompressed) throws IOException {
    boolean isCodeCell = false;
    Optional<NotebookParsingData> notebookData = Optional.empty();
    while (jParser.nextToken() != JsonToken.END_OBJECT) {

      skipNestedObjects(jParser);

      if (!isCodeCell) {
        isCodeCell = processCodeCellType(jParser);
      }

      if (JsonToken.FIELD_NAME.equals(jParser.currentToken()) && "source".equals(jParser.currentName())) {
        jParser.nextToken();

        int startLine = 0;
        if (!accumulator.isEmpty()) {
          startLine = accumulator.get(accumulator.size() - 1).getAggregatedSourceLine();
        }
        switch (jParser.currentToken()) {
          case START_ARRAY:
            notebookData = Optional.of(parseSourceArray(startLine, jParser, isCompressed));
            break;
          case VALUE_STRING:
            notebookData = Optional.of(parseSourceMultilineString(startLine, jParser));
            break;
          default:
            throw new IllegalStateException("Unexpected token: " + jParser.currentToken());
        }
      }
    }

    if (isCodeCell && notebookData.isPresent()) {
      var data = notebookData.get();
      lastPythonLine = data.getAggregatedSourceLine();
      accumulator.add(data);
    }
  }

  private static NotebookParsingData parseSourceArray(int startLine, JsonParser jParser, boolean isCompressed) throws IOException {
    NotebookParsingData cellData = NotebookParsingData.fromLine(startLine);
    JsonLocation tokenLocation = jParser.currentTokenLocation();
    // In case of an empty cell, we don't add an extra line
    var lastSourceLine = "\n";
    while (jParser.nextToken() != JsonToken.END_ARRAY) {
      String sourceLine = jParser.getValueAsString();
      var newTokenLocation = jParser.currentTokenLocation();
      var countEscapedChar = countEscapeCharacters(sourceLine);
      cellData.addLineToSource(sourceLine, newTokenLocation.getLineNr(), newTokenLocation.getColumnNr(), countEscapedChar, isCompressed);
      lastSourceLine = sourceLine;
      tokenLocation = newTokenLocation;
    }
    if (!lastSourceLine.endsWith("\n")) {
      cellData.appendToSource("\n");
    }
    // Account for the last cell delimiter
    cellData.addDelimiterToSource(SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER + "\n", tokenLocation.getLineNr(), tokenLocation.getColumnNr());
    return cellData;
  }

  private static NotebookParsingData parseSourceMultilineString(int startLine, JsonParser jParser) throws IOException {
    NotebookParsingData cellData = NotebookParsingData.fromLine(startLine);
    String sourceLine = jParser.getValueAsString();
    JsonLocation tokenLocation = jParser.currentTokenLocation();
    var previousLen = 0;
    var previousExtraChars = 0;

    for (String line : sourceLine.lines().toList()) {
      var countEscapedChar = countEscapeCharacters(line);
      var currentCount = countEscapedChar.stream().mapToInt(EscapeCharPositionInfo::numberOfExtraChars).sum();
      cellData.addLineToSource(line, new IPythonLocation(tokenLocation.getLineNr(),
        tokenLocation.getColumnNr() + previousLen + previousExtraChars, countEscapedChar, true));
      cellData.appendToSource("\n");
      previousLen += line.length() + 2;
      previousExtraChars += currentCount;
    }
    // Account for the last cell delimiter
    cellData.addDelimiterToSource(SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER + "\n", tokenLocation.getLineNr(), tokenLocation.getColumnNr());
    return cellData;
  }

  private static List<EscapeCharPositionInfo> countEscapeCharacters(String sourceLine) {
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
}
