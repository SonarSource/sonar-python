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

public class IpynbNotebookParser {

  public static final String SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER = "\n#SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER\n";

  public static ParseResult parseNotebook(InputFile inputFile) {
    try {
      return new InternalParser(inputFile).parseNotebook();
    } catch (IOException e) {
      throw new IllegalStateException("Cannot read " + inputFile, e);
    }
  }

  private static class InternalParser {
    private InputFile inputFile;
    private StringBuilder aggregatedSource = new StringBuilder();

    // Keys are the aggregated source line number
    private final Map<Integer, JsonLocation> locationMap = new HashMap<>();
    private final Map<Integer, Offset> offSetMap = new HashMap<>();
    private int aggregatedSourceLine = 1;

    private InternalParser(InputFile inputFile) {
      this.inputFile = inputFile;
    }

    public ParseResult parseNotebook() throws IOException {
      String content = inputFile.contents();
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

      return new ParseResult(inputFile, aggregatedSource.toString(), offSetMap);
    }

    private void processCodeCell(JsonParser jParser) throws IOException {
      while (!jParser.isClosed()) {
        JsonToken jsonToken = jParser.nextToken();
        if (JsonToken.FIELD_NAME.equals(jsonToken) && "source".equals(jParser.currentName())) {
          jsonToken = jParser.nextToken();
          if (jsonToken == JsonToken.START_ARRAY) {
            while (jParser.nextToken() != JsonToken.END_ARRAY) {
              String sourceLine = jParser.getValueAsString();
              JsonLocation tokenLocation = jParser.currentTokenLocation();

              aggregatedSource.append(sourceLine);
              locationMap.put(aggregatedSourceLine, tokenLocation);
              offSetMap.put(aggregatedSourceLine, new Offset(tokenLocation.getLineNr(), tokenLocation.getColumnNr()));
              aggregatedSourceLine++;
            }
            // Account for the cell delimiter
            aggregatedSource.append(SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER);
            aggregatedSourceLine++;
            break;
          }
        }
      }
    }
  }

  public record ParseResult(InputFile inputFile, String aggregatedSource, Map<Integer, Offset> offsetMap) {

  }

  public record Offset(int line, int column) {
  }

}
