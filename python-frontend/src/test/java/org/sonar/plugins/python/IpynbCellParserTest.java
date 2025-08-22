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

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import java.io.IOException;
import java.io.StringReader;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.sonar.python.IPythonLocation;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class IpynbCellParserTest {

  private final JsonFactory jsonFactory = new JsonFactory();
  private final IpynbCellParser parser = new IpynbCellParser();

  @Test
  void identifiesCodeCell() throws IOException {
    String json = """
      {
        "cell_type": "code",
        "source": ["print('hello')"]
      }
      """;

    JsonParser jParser = getJsonObjectParser(json);

    parser.processCodeCell(jParser);
    var cellData = parser.getCellData();

    assertThat(cellData).hasSize(1);
    assertThat(cellData.get(0).getAggregatedSource().toString())
      .contains("print('hello')")
      .contains(IpynbCellParser.SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER);
  }

  @Test
  void emptySource() throws IOException {
    String json = """
      {
        "cell_type": "code",
        "source": []
      }
      """;

    JsonParser jParser = getJsonObjectParser(json);

    parser.processCodeCell(jParser);
    var accumulator = parser.getCellData();

    assertThat(accumulator).hasSize(1);
    NotebookParsingData data = accumulator.get(0);
    assertThat(data.getAggregatedSource())
      .hasToString(IpynbCellParser.SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER + "\n");
  }

  @Test
  void skipsNonCodeCells() throws IOException {
    String json = """
      {
        "cell_type": "markdown",
        "source": ["# This is markdown"]
      }
      """;

    JsonParser jParser = getJsonObjectParser(json);

    parser.processCodeCell(jParser);
    var accumulator = parser.getCellData();
    assertThat(accumulator).isEmpty();
  }

  @Test
  void handlesMissingCellType() throws IOException {
    String json = """
      {
        "source": ["print('hello')"]
      }
      """;

    JsonParser jParser = getJsonObjectParser(json);

    parser.processCodeCell(jParser);
    var accumulator = parser.getCellData();
    assertThat(accumulator).isEmpty();
  }

  private static Stream<Arguments> lineEndings() {
    return Stream.of(
      Arguments.of("""
        {
          "cell_type": "code",
          "source": [
            "x = 42\\n",
            "\\n",
            "print(x)\\n"
          ]
        }
        """),
      Arguments.of("""
        {
          "cell_type": "code",
          "source": [
            "x = 42",
            "",
            "print(x)",
            ""
          ]
        }
        """),
      Arguments.of("""
        {
          "cell_type": "code",
          "source": [
            "x = 42\\n",
            "\\n",
            "print(x)\\n",
            ""
          ]
        }
        """));
  }

  @ParameterizedTest
  @MethodSource("lineEndings")
  void parsesArrayEndingWithNewLine(String json) throws IOException {
    JsonParser jParser = getJsonObjectParser(json);

    parser.processCodeCell(jParser);
    var accumulator = parser.getCellData();

    assertThat(accumulator).hasSize(1);
    NotebookParsingData data = accumulator.get(0);
    assertThat(data.getAggregatedSource())
      .hasToString("""
        x = 42

        print(x)

        #SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER
        """);
    assertThat(data.getLocationMap()).hasSize(5); // 4 lines + delimiter
  }

  @Test
  void parsesEmptyString() throws IOException {
    String json = """
      {
        "cell_type": "code",
        "source": ""
      }
      """;

    JsonParser jParser = getJsonObjectParser(json);

    parser.processCodeCell(jParser);
    var accumulator = parser.getCellData();

    assertThat(accumulator).hasSize(1);
    NotebookParsingData data = accumulator.get(0);
    assertThat(data.getAggregatedSource())
      .hasToString("#SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER\n");
  }

  @Test
  void parsesSingleStringSource() throws IOException {
    String json = """
      {
        "cell_type": "code",
        "source": "print('hello world')"
      }
      """;

    JsonParser jParser = getJsonObjectParser(json);

    parser.processCodeCell(jParser);
    var accumulator = parser.getCellData();

    assertThat(accumulator).hasSize(1);
    NotebookParsingData data = accumulator.get(0);
    assertThat(data.getAggregatedSource())
      .hasToString("""
        print('hello world')
        #SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER
        """);
  }

  @Test
  void parsesMultilineString() throws IOException {
    String json = """
      {
        "cell_type": "code",
        "source": "x = 42\\nprint(x)\\nprint('done')"
      }
      """;

    JsonParser jParser = getJsonObjectParser(json);
    parser.processCodeCell(jParser);

    List<NotebookParsingData> cellData = parser.getCellData();
    assertThat(cellData).hasSize(1);
    NotebookParsingData data = cellData.get(0);
    assertThat(data.getAggregatedSource())
      .hasToString("""
        x = 42
        print(x)
        print('done')
        #SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER
        """);
  }

  @Test
  void parsesMultilineStringEndingWithNewLine() throws IOException {
    String json = """
      {
        "cell_type": "code",
        "source": "x = 42\\nprint(x)\\nprint('done')\\n"
      }
      """;

    JsonParser jParser = getJsonObjectParser(json);

    parser.processCodeCell(jParser);
    List<NotebookParsingData> cellData = parser.getCellData();
    assertThat(cellData).hasSize(1);
    NotebookParsingData data = cellData.get(0);
    assertThat(data.getAggregatedSource())
      .hasToString("""
        x = 42
        print(x)
        print('done')

        #SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER
        """);
  }

  @Test
  void handlesEscapeCharacters() throws IOException {
    String json = """
      {
        "cell_type": "code",
        "source": ["print(\\"hello\\\\tworld\\")\\n"]
      }
      """;

    JsonParser jParser = getJsonObjectParser(json);

    parser.processCodeCell(jParser);
    var accumulator = parser.getCellData();

    assertThat(accumulator).hasSize(1);
    NotebookParsingData data = accumulator.get(0);
    assertThat(data.getAggregatedSource().toString()).contains("print(\"hello\\tworld\")");

    // Verify escape character positions are tracked
    IPythonLocation location = data.getLocationMap().get(1);
    assertThat(location.colOffsets()).hasSize(3);
  }

  @Test
  void tracksLineNumbers() throws IOException {
    String json = """
      {
        "cell_type": "code",
        "source": [
          "line1\\n",
          "line2\\n"
        ]
      }
      """;

    JsonParser jParser = getJsonObjectParser(json);

    parser.processCodeCell(jParser);
    var accumulator = parser.getCellData();

    assertThat(accumulator).hasSize(1);
    NotebookParsingData data = accumulator.get(0);
    Map<Integer, IPythonLocation> locationMap = data.getLocationMap();

    assertThat(locationMap).containsKeys(1, 2, 3); // 2 lines + delimiter
    assertThat(locationMap.get(1).line()).isEqualTo(4);
    assertThat(locationMap.get(2).line()).isEqualTo(5);
  }

  @Test
  void skipsNestedObjects() throws IOException {
    String json = """
      {
        "cell_type": "code",
        "metadata": {
          "nested": {
            "deep": "value"
          }
        },
        "outputs": [
          {
            "output_type": "stream",
            "text": ["some output"]
          }
        ],
        "source": ["print('hello')"]
      }
      """;

    JsonParser jParser = getJsonObjectParser(json);

    parser.processCodeCell(jParser);
    var accumulator = parser.getCellData();

    assertThat(accumulator).hasSize(1);
    assertThat(accumulator.get(0).getAggregatedSource())
      .hasToString("""
        print('hello')
        #SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER
        """);
  }

  @Test
  void invalidJsonStructure() throws IOException {
    String json = """
      {
        "cell_type": "code",
        "source": 42
      }
      """;

    JsonParser jParser = getJsonObjectParser(json);

    assertThatThrownBy(() -> parser.processCodeCell(jParser))
      .isInstanceOf(IllegalStateException.class)
      .hasMessageContaining("Unexpected token");
  }

  private JsonParser getJsonObjectParser(String json) throws IOException {
    JsonParser jParser = jsonFactory.createParser(new StringReader(json));
    jParser.nextToken(); // Move to START_OBJECT
    return jParser;
  }
}
