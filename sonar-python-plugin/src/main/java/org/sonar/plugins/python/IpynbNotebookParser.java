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
import java.util.Map;

public class IpynbNotebookParser {

  public static final String SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER = "\n#SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER\n";

  public static ParseResult parseNotebook(PythonInputFile inputFile) {
    try {
      return new IpynbNotebookParser(inputFile).parseNotebook();
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
  private int aggregatedSourceLine = 1;

  public ParseResult parseNotebook() throws IOException {
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
    }

    return new ParseResult(inputFile, aggregatedSource.toString(), locationMap);
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

  private boolean parseSourceArray(JsonParser jParser, JsonToken jsonToken) throws IOException {
    if (jsonToken != JsonToken.START_ARRAY) {
      return false;
    }
    while (jParser.nextToken() != JsonToken.END_ARRAY) {
      String sourceLine = jParser.getValueAsString();
      JsonLocation tokenLocation = jParser.currentTokenLocation();

      aggregatedSource.append(sourceLine);
      locationMap.put(aggregatedSourceLine, new IPythonLocation(tokenLocation.getLineNr(), tokenLocation.getColumnNr()));
      aggregatedSourceLine++;
    }
    // Account for the last cell delimiter
    aggregatedSource.append(SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER);
    aggregatedSourceLine++;
    return true;
  }

  private boolean parseSourceMultilineString(JsonParser jParser, JsonToken jsonToken) throws IOException {
    if (jsonToken != JsonToken.VALUE_STRING) {
      return false;
    }
    String sourceLine = jParser.getValueAsString();
    JsonLocation tokenLocation = jParser.currentTokenLocation();
    var jsonColOffset = 1;

    for (String line : sourceLine.lines().toList()) {
      aggregatedSource.append(line);
      aggregatedSource.append("\n");
      locationMap.put(aggregatedSourceLine, new IPythonLocation(tokenLocation.getLineNr(), tokenLocation.getColumnNr() + jsonColOffset));
      jsonColOffset += line.length() + 2 + countEscapeCharacters(line);
      aggregatedSourceLine++;
    }
    // Account for the last cell delimiter
    aggregatedSource.append(SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER);
    aggregatedSourceLine++;
    return true;
  }

  private static int countEscapeCharacters(String str) {
    int count = 0;
    for (char c : str.toCharArray()) {
      if (c == '"' || c == '\\') {
        count++;
      }
    }
    return count;
  }

  public record ParseResult(PythonInputFile inputFile, String aggregatedSource, Map<Integer, IPythonLocation> locationMap) {
  }

  public record IPythonLocation(int line, int column) {
  }

}
