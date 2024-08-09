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
import org.sonar.api.batch.fs.InputFile;

public class IPythonNotebookParser {

  public static GeneratedIPythonFile parseNotebook(InputFile inputFile) throws IOException {
    String content = inputFile.contents();
    JsonFactory factory = new JsonFactory();
    try (JsonParser parser = factory.createParser(content)) {
      while (!parser.isClosed()) {
        JsonToken jsonToken = parser.nextToken();
        if (JsonToken.FIELD_NAME.equals(jsonToken)) {
          String fieldName = parser.getCurrentName();
          if ("cell_type".equals(fieldName)) {
            parser.nextToken();
            if ("code".equals(parser.getValueAsString())) {
              processCodeCell(parser);
            }
          }
        }
      }
    }
    System.out.println(aggregatedSource);
    return new GeneratedIPythonFile(inputFile, aggregatedSource.toString(), offSetMap);
  }

  private static StringBuilder aggregatedSource = new StringBuilder();
  private static Map<Integer, JsonLocation> locationMap = new HashMap<>();
  private static Map<Integer, GeneratedIPythonFile.Offset> offSetMap = new HashMap<>();
  private static int aggregatedSourceLine = 1;

  private static void processCodeCell(JsonParser parser) throws IOException {
    while (!parser.isClosed()) {
      JsonToken jsonToken = parser.nextToken();
      if (JsonToken.FIELD_NAME.equals(jsonToken) && "source".equals(parser.getCurrentName())) {
        jsonToken = parser.nextToken();
        if (jsonToken == JsonToken.START_ARRAY) {
          while (parser.nextToken() != JsonToken.END_ARRAY) {
            String sourceLine = parser.getValueAsString();
            JsonLocation tokenLocation = parser.getTokenLocation();
            aggregatedSource.append(sourceLine);
            locationMap.put(aggregatedSourceLine, tokenLocation);
            offSetMap.put(aggregatedSourceLine, new GeneratedIPythonFile.Offset(tokenLocation.getLineNr(), tokenLocation.getColumnNr()));
            aggregatedSourceLine++;
          }
          // account for the cell delimiter
          aggregatedSource.append("\n#SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER\n");
          aggregatedSourceLine++;
          break;
        }
      }
    }
  }
}
