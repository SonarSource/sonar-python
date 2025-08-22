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
import com.fasterxml.jackson.core.JsonLocation;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Optional;
import java.util.Set;

public class IpynbNotebookParser {

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
        notebookData.addDefaultLocation(notebookData.getAggregatedSourceLine(), location.getLineNr(), location.getColumnNr());
        return new GeneratedIPythonFile(inputFile.wrappedFile(), notebookData.getAggregatedSource().toString(), notebookData.getLocationMap());
      }).orElse(new GeneratedIPythonFile(inputFile.wrappedFile(), "", new LinkedHashMap<>()));
    }

  }

  private static Optional<NotebookParsingData> parseCells(JsonParser parser) throws IOException {
    while (!parser.isClosed()) {
      parser.nextToken();
      String fieldName = parser.currentName();
      if ("cells".equals(fieldName)) {
        // consume array start token
        parser.nextToken();
        Optional<NotebookParsingData> data = parseCellArray(parser);
        parser.close();
        return data;
      }
    }
    return Optional.empty();
  }

  private static Optional<NotebookParsingData> parseCellArray(JsonParser jParser) throws IOException {
    IpynbCellParser cellParser = new IpynbCellParser();

    while (jParser.nextToken() != JsonToken.END_ARRAY) {
      if (jParser.currentToken() == JsonToken.START_OBJECT) {
        cellParser.processCodeCell(jParser);
      }
    }

    Optional<NotebookParsingData> aggregatedNotebookData = cellParser.getCellData().stream().reduce(NotebookParsingData::combine);
    aggregatedNotebookData.ifPresent(NotebookParsingData::removeTrailingExtraLine);
    return aggregatedNotebookData;
  }
}
