/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
package org.sonar.plugins.python;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonLocation;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
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
    JsonFactory factory = new JsonFactory();
    try (JsonParser jParser = factory.createParser(content)) {
      return parseCells(jParser).map(notebookData -> {
        // Account for EOF token
        JsonLocation location = jParser.currentTokenLocation();
        notebookData.addDefaultLocation(lastPythonLine, location.getLineNr(), location.getColumnNr());
        return new GeneratedIPythonFile(inputFile.wrappedFile(), notebookData.getAggregatedSource().toString(), notebookData.getLocationMap());
      }).orElse(new GeneratedIPythonFile(inputFile.wrappedFile(), "", new LinkedHashMap<>()));
    }

  }

  private Optional<NotebookParsingData> parseCells(JsonParser parser) throws IOException {
    while (!parser.isClosed()) {
      parser.nextToken();
      String fieldName = parser.currentName();
      if ("cells".equals(fieldName)) {
        // consume array start token
        parser.nextToken();
        NotebookParsingData data = parseCellArray(parser);
        parser.close();
        return Optional.of(data);
      }
    }
    return Optional.empty();
  }

  private NotebookParsingData parseCellArray(JsonParser jParser) throws IOException {
    NotebookParsingData aggregatedNotebookData = NotebookParsingData.empty();

    while (jParser.nextToken() != JsonToken.END_ARRAY) {
      if (jParser.currentToken() == JsonToken.START_OBJECT) {
        processCodeCell(aggregatedNotebookData.getAggregatedSourceLine(), jParser).ifPresent(aggregatedNotebookData::combine);
      }
    }
    aggregatedNotebookData.removeTrailingExtraLine();
    return aggregatedNotebookData;
  }

  private static void skipNestedObjects(JsonParser parser) throws IOException {
    if (parser.currentToken() == JsonToken.START_OBJECT || parser.currentToken() == JsonToken.START_ARRAY) {
      parser.skipChildren();
    }
  }

  private Optional<NotebookParsingData> processCodeCell(int startLine, JsonParser jParser) throws IOException {
    boolean isCodeCell = false;
    Optional<NotebookParsingData> notebookData = Optional.empty();
    while (jParser.nextToken() != JsonToken.END_OBJECT) {

      skipNestedObjects(jParser);

      if (JsonToken.FIELD_NAME.equals(jParser.currentToken()) && "cell_type".equals(jParser.currentName())) {
        jParser.nextToken();
        String cellType = jParser.getValueAsString();
        if ("code".equals(cellType)) {
          isCodeCell = true;
        }
      }
      if (JsonToken.FIELD_NAME.equals(jParser.currentToken()) && "source".equals(jParser.currentName())) {
        jParser.nextToken();
        switch (jParser.currentToken()) {
          case START_ARRAY:
            notebookData = Optional.of(parseSourceArray(startLine, jParser));
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
      lastPythonLine = notebookData.get().getAggregatedSourceLine();
      return notebookData;
    }
    return Optional.empty();
  }


  private static NotebookParsingData parseSourceArray(int startLine, JsonParser jParser) throws IOException {
    NotebookParsingData cellData = NotebookParsingData.fromLine(startLine);
    JsonLocation tokenLocation = jParser.currentTokenLocation();
    // In case of an empty cell, we don't add an extra line
    var lastSourceLine = "\n";
    while (jParser.nextToken() != JsonToken.END_ARRAY) {
      String sourceLine = jParser.getValueAsString();
      tokenLocation = jParser.currentTokenLocation();
      var countEscapedChar = countEscapeCharacters(sourceLine, tokenLocation.getColumnNr());
      cellData.addLineToSource(sourceLine, tokenLocation.getLineNr(), tokenLocation.getColumnNr(), countEscapedChar);
      lastSourceLine = sourceLine;
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
      var countEscapedChar = countEscapeCharacters(line, previousLen + previousExtraChars + tokenLocation.getColumnNr());
      var currentCount = countEscapedChar.get(-1);
      cellData.addLineToSource(line, new IPythonLocation(tokenLocation.getLineNr(),
        tokenLocation.getColumnNr() + previousLen + previousExtraChars, countEscapedChar));
      cellData.appendToSource("\n");
      previousLen = previousLen + line.length() + 2;
      previousExtraChars = previousExtraChars + currentCount;
    }
    // Account for the last cell delimiter
    cellData.addDelimiterToSource(SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER + "\n", tokenLocation.getLineNr(), tokenLocation.getColumnNr());
    return cellData;
  }

  private static Map<Integer, Integer> countEscapeCharacters(String sourceLine, int colOffSet) {
    Map<Integer, Integer> colMap = new LinkedHashMap<>();
    int count = 0;
    var numberOfExtraChars = 0;
    var arr = sourceLine.toCharArray();
    for (int i = 1; i < sourceLine.length(); ++i) {
      char c = arr[i];
      switch (c) {
        case '"', '\\':
          numberOfExtraChars++;
          colMap.put(i, i + colOffSet + count + numberOfExtraChars);
          break;
        // we never encounter \n or \r as the lines are split at these characters 
        case '\b', '\f', '\t':
          // we increase the count of one char as we count the \ but not the t or b
          count += 1;
          break;
        default:
          break;
      }
    }
    colMap.put(-1, numberOfExtraChars);
    return colMap;
  }
}
