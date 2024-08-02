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
import java.util.HashMap;
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
  private final StringBuilder aggregatedSource = new StringBuilder();

  // Keys are the aggregated source line number
  private final Map<Integer, IPythonLocation> locationMap = new HashMap<>();
  private int aggregatedSourceLine = 0;
  private int lastPythonLine = 0;
  private boolean isFirstCell = true;

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
        if (JsonToken.FIELD_NAME.equals(jsonToken)) {
          String fieldName = jParser.currentName();
          if ("language".equals(fieldName)) {
            jParser.nextToken();
            return Optional.ofNullable(jParser.getValueAsString());
          }
        }
      }
    }
    return Optional.empty();
  }

  public GeneratedIPythonFile parseNotebook() throws IOException {
    String content = inputFile.wrappedFile().contents();
    JsonFactory factory = new JsonFactory();
    try (JsonParser jParser = factory.createParser(content)) {
      while (!jParser.isClosed()) {
        JsonToken jsonToken = jParser.nextToken();
        if (JsonToken.FIELD_NAME.equals(jsonToken)) {
          String fieldName = jParser.currentName();
          if ("cell_type".equals(fieldName)) {
            jParser.nextToken();
            if ("code".equals(jParser.getValueAsString())) {
              processCodeCell(jParser);
            }
          }
        }
      }
      // Account for EOF token
      addDefaultLocation(lastPythonLine, jParser.currentTokenLocation());
    }

    return new GeneratedIPythonFile(inputFile.wrappedFile(), aggregatedSource.toString(), locationMap);
  }

  private void processCodeCell(JsonParser jParser) throws IOException {

    while (!jParser.isClosed()) {
      JsonToken jsonToken = jParser.nextToken();
      if (JsonToken.FIELD_NAME.equals(jsonToken) && "source".equals(jParser.currentName())) {
        jsonToken = jParser.nextToken();
        if (parseSourceArray(jParser, jsonToken) || parseSourceMultilineString(jParser, jsonToken)) {
          break;
        } else {
          throw new IllegalStateException("Unexpected token: " + jsonToken);
        }
      }
    }
  }

  private void appendNewLineAfterPreviousCellDelimiter() {
    if (!isFirstCell) {
      aggregatedSource.append("\n");
    } else {
      isFirstCell = false;
    }
  }

  private boolean parseSourceArray(JsonParser jParser, JsonToken jsonToken) throws IOException {
    if (jsonToken != JsonToken.START_ARRAY) {
      return false;
    }
    appendNewLineAfterPreviousCellDelimiter();
    JsonLocation tokenLocation = jParser.currentTokenLocation();
    while (jParser.nextToken() != JsonToken.END_ARRAY) {
      String sourceLine = jParser.getValueAsString();
      tokenLocation = jParser.currentTokenLocation();
      var countEscapedChar = countEscapeCharacters(sourceLine, new LinkedHashMap<>(), tokenLocation.getColumnNr());
      addLineToSource(sourceLine, tokenLocation, countEscapedChar);
    }
    aggregatedSource.append("\n");
    // Account for the last cell delimiter
    addDelimiterToSource(tokenLocation);
    lastPythonLine = aggregatedSourceLine;
    return true;
  }

  private boolean parseSourceMultilineString(JsonParser jParser, JsonToken jsonToken) throws IOException {
    if (jsonToken != JsonToken.VALUE_STRING) {
      return false;
    }
    appendNewLineAfterPreviousCellDelimiter();
    String sourceLine = jParser.getValueAsString();
    JsonLocation tokenLocation = jParser.currentTokenLocation();
    var previousLen = 0;
    var previousExtraChars = 0;

    for (String line : sourceLine.lines().toList()) {
      var countEscapedChar = countEscapeCharacters(line, new LinkedHashMap<>(), previousLen + previousExtraChars + tokenLocation.getColumnNr());
      var currentCount = countEscapedChar.get(-1);
      addLineToSource(line, new IPythonLocation(tokenLocation.getLineNr(),
        tokenLocation.getColumnNr() + previousLen + previousExtraChars, countEscapedChar));
      aggregatedSource.append("\n");
      previousLen = line.length() + 2;
      previousExtraChars = currentCount;
    }
    // Account for the last cell delimiter
    addDelimiterToSource(tokenLocation);
    lastPythonLine = aggregatedSourceLine;
    return true;
  }

  private void addLineToSource(String sourceLine, JsonLocation tokenLocation, Map<Integer, Integer> colOffset) {
    addLineToSource(sourceLine, new IPythonLocation(tokenLocation.getLineNr(), tokenLocation.getColumnNr(), colOffset));
  }

  private void addLineToSource(String sourceLine, IPythonLocation location) {
    aggregatedSource.append(sourceLine);
    aggregatedSourceLine++;
    locationMap.put(aggregatedSourceLine, location);
  }

  private void addDelimiterToSource(JsonLocation tokenLocation) {
    aggregatedSource.append(SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER);
    aggregatedSourceLine++;
    addDefaultLocation(aggregatedSourceLine, tokenLocation);
  }

  private void addDefaultLocation(int line, JsonLocation tokenLocation) {
    locationMap.putIfAbsent(line, new IPythonLocation(tokenLocation.getLineNr(), tokenLocation.getColumnNr(), Map.of(-1, 0)));
  }

  private static Map<Integer, Integer> countEscapeCharacters(String sourceLine, Map<Integer, Integer> colMap, int colOffSet) {
    int count = 0;
    var numberOfExtraChars = 0;
    var arr = sourceLine.toCharArray();
    for (int i = 1; i < sourceLine.length(); ++i) {
      char c = arr[i];
      switch (c) {
        case '"', '\'', '\\':
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
