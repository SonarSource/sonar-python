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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.python.EscapeCharPositionInfo;
import org.sonar.python.IPythonLocation;

public class IpynbNotebookParser {

  private static final Logger LOG = LoggerFactory.getLogger(IpynbNotebookParser.class);

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
    var language = parseLanguage();
    boolean isPythonNotebook = language.map(ACCEPTED_LANGUAGE::contains).orElse(true);

    if (isPythonNotebook) {
      return Optional.of(parseNotebook());
    }

    if(LOG.isDebugEnabled()){
      LOG.debug("Skipping notebook '{}': unsupported language '{}'", inputFile.wrappedFile().filename(), language.orElse("unknown"));
    }
    return Optional.empty();
  }

  /**
   * Parses the notebook's top-level metadata to find the language.
   * Only checks metadata.kernelspec.language and metadata.language_info.name,
   * ignoring any language fields in cell metadata.
   */
  public Optional<String> parseLanguage() throws IOException {
    String content = inputFile.wrappedFile().contents();
    JsonFactory factory = new JsonFactory();
    List<String> foundLanguages = new ArrayList<>();

    try (JsonParser jParser = factory.createParser(content)) {
      while (!jParser.isClosed()) {
        JsonToken jsonToken = jParser.nextToken();
        if (JsonToken.FIELD_NAME.equals(jsonToken) && "metadata".equals(jParser.currentName()) && jParser.getParsingContext().getParent().inRoot()) {
          jParser.nextToken();
          extractLanguagesFromMetadata(jParser, foundLanguages);
          break;
        }
      }
    }

    // Return an accepted language if found, otherwise the first language found (for rejection), or empty
    return foundLanguages.stream()
      .filter(ACCEPTED_LANGUAGE::contains)
      .findFirst()
      .or(() -> foundLanguages.stream().findFirst());
  }

  /**
   * Extracts language values from the top-level metadata object.
   * Looks for kernelspec.language and language_info.name.
   */
  private static void extractLanguagesFromMetadata(JsonParser jParser, List<String> foundLanguages) throws IOException {
    while (jParser.nextToken() != JsonToken.END_OBJECT) {
      if (JsonToken.FIELD_NAME.equals(jParser.currentToken())) {
        String fieldName = jParser.currentName();
        if ("kernelspec".equals(fieldName)) {
          jParser.nextToken();
          extractFieldFromObject(jParser, "language", foundLanguages);
        } else if ("language_info".equals(fieldName)) {
          jParser.nextToken();
          extractFieldFromObject(jParser, "name", foundLanguages);
        } else {
          jParser.nextToken();
          skipNestedObjects(jParser);
        }
      }
    }
  }

  /**
   * Extracts the value of a specific field from a JSON object.
   */
  private static void extractFieldFromObject(JsonParser jParser, String targetField, List<String> foundValues) throws IOException {
    while (jParser.nextToken() != JsonToken.END_OBJECT) {
      if (JsonToken.FIELD_NAME.equals(jParser.currentToken()) && targetField.equals(jParser.currentName())) {
        jParser.nextToken();
        String value = jParser.getValueAsString();
        if (value != null) {
          foundValues.add(value);
        }
      } else {
        jParser.nextToken();
        skipNestedObjects(jParser);
      }
    }
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
    var lastOffset = LineSplitOffset.NONE;
    // Whether the array element about to be processed starts on a fresh physical line. An element only
    // does if the previous one ended with a newline; otherwise it is glued onto the previous element's
    // still-open line and must not get its own locationMap entry (there is only room for one per line).
    var startsNewLine = true;
    while (jParser.nextToken() != JsonToken.END_ARRAY) {
      String sourceLine = jParser.getValueAsString();
      var newTokenLocation = jParser.currentTokenLocation();
      lastOffset = addSourceArrayElement(cellData, sourceLine, newTokenLocation, isCompressed, startsNewLine);
      startsNewLine = sourceLine.endsWith("\n");
      lastSourceLine = sourceLine;
      tokenLocation = newTokenLocation;
    }
    // Column right after the last array element's content, accounting for any lines split out of a
    // multiline string contained in that element.
    int lastColumn = tokenLocation.getColumnNr() + lastOffset.length() + lastOffset.extraChars();
    if (!lastSourceLine.endsWith("\n")) {
      cellData.appendToSource("\n");
    } else {
      // if the last string of the array ends with a newline character we should add this new line to our representation
      var newLineLocation = new IPythonLocation(
        tokenLocation.getLineNr(),
        lastColumn,
        List.of(new EscapeCharPositionInfo(lastColumn, 1)),
        false);
      cellData.addLineToSource("\n", newLineLocation);
    }
    // Account for the last cell delimiter
    cellData.addDelimiterToSource(SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER + "\n", tokenLocation.getLineNr(), lastColumn);
    return cellData;
  }

  /**
   * Adds a single "source" array element to the cell data. Most array elements represent exactly one
   * Python source line, but the JSON schema also allows an element to itself contain embedded newlines
   * (a multiline string used as one array item), in which case it is split the same way a top-level
   * multiline string "source" value is. Returns the raw-content offset consumed within the element's
   * JSON token, relative to its token location, so the caller can correctly position whatever follows it.
   *
   * <p>If the previous element did not end with a newline, this element continues that still-open
   * physical line rather than starting a new one: its first (or only) segment is appended as plain text,
   * with no locationMap entry of its own, since a generated line can only be anchored to a single
   * original position.
   */
  private static LineSplitOffset addSourceArrayElement(NotebookParsingData cellData, String sourceLine, JsonLocation tokenLocation, boolean isCompressed,
    boolean startsNewLine) {
    List<String> lines = sourceLine.lines().toList();
    if (lines.size() <= 1) {
      if (!startsNewLine) {
        cellData.appendToSource(sourceLine);
        return LineSplitOffset.NONE;
      }
      var countEscapedChar = countEscapeCharacters(sourceLine);
      cellData.addLineToSource(sourceLine, tokenLocation.getLineNr(), tokenLocation.getColumnNr(), countEscapedChar, isCompressed);
      return LineSplitOffset.NONE;
    }
    // The element packs multiple Python lines into a single array entry: each embedded line needs its
    // own location entry, the same way parseSourceMultilineString handles a plain-string "source" -
    // except its first line, which only gets one if it genuinely starts a new physical line.
    var offset = addSourceLinesToCellData(cellData, lines, tokenLocation, true, startsNewLine);
    if (sourceLine.endsWith("\n")) {
      cellData.appendToSource("\n");
      offset = offset.plusNewline();
    }
    return offset;
  }

  private static NotebookParsingData parseSourceMultilineString(int startLine, JsonParser jParser) throws IOException {
    NotebookParsingData cellData = NotebookParsingData.fromLine(startLine);
    String sourceLine = jParser.getValueAsString();
    JsonLocation tokenLocation = jParser.currentTokenLocation();

    List<String> lines = sourceLine.lines().toList();
    var offset = addSourceLinesToCellData(cellData, lines, tokenLocation, true, true);
    // The last split line is always followed by a newline: either the cell delimiter or the next cell's content.
    // An empty source ("") has no split line to terminate, so there is nothing to append here - doing so
    // regardless would insert a physical blank line with no locationMap entry of its own, drifting every
    // following location by one.
    if (!lines.isEmpty()) {
      cellData.appendToSource("\n");
      offset = offset.plusNewline();
    }

    if (sourceLine.endsWith("\n")) {
      var column = tokenLocation.getColumnNr() + offset.length() + offset.extraChars();
      cellData.addLineToSource("\n", new IPythonLocation(tokenLocation.getLineNr(), column, List.of(new EscapeCharPositionInfo(column, 1)), true));
      offset = offset.plusNewline();
    }
    // Account for the last cell delimiter
    cellData.addDelimiterToSource(SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER + "\n", tokenLocation.getLineNr(), tokenLocation.getColumnNr() + offset.length() + offset.extraChars());
    return cellData;
  }

  /**
   * Splits a JSON string value's content into individual Python source lines and adds each to the cell
   * data, computing the column of every line within the original JSON token. Every line except the last
   * is followed by an explicit newline in the aggregated source, since it was followed by an embedded
   * newline in the JSON value; the caller decides how to terminate the last one.
   *
   * <p>When {@code firstLineStartsNewLine} is false, the first line continues a still-open physical line
   * from whatever was appended just before it, so it is added as plain text with no locationMap entry of
   * its own (a generated line can only be anchored to a single original position).
   */
  private static LineSplitOffset addSourceLinesToCellData(NotebookParsingData cellData, List<String> lines, JsonLocation tokenLocation, boolean isCompressed,
    boolean firstLineStartsNewLine) {
    var offset = LineSplitOffset.NONE;
    for (int i = 0; i < lines.size(); i++) {
      String line = lines.get(i);
      var countEscapedChar = countEscapeCharacters(line);
      var currentExtraChars = countEscapedChar.stream().mapToInt(EscapeCharPositionInfo::numberOfExtraChars).sum();
      if (i > 0 || firstLineStartsNewLine) {
        cellData.addLineToSource(line, new IPythonLocation(tokenLocation.getLineNr(),
          tokenLocation.getColumnNr() + offset.length() + offset.extraChars(), countEscapedChar, isCompressed));
      } else {
        cellData.appendToSource(line);
      }
      boolean hasMoreLines = i < lines.size() - 1;
      if (hasMoreLines) {
        cellData.appendToSource("\n");
        offset = offset.plusLine(line.length(), currentExtraChars);
      } else {
        offset = offset.plusLastLine(line.length(), currentExtraChars);
      }
    }
    return offset;
  }

  /**
   * Tracks how many raw JSON characters (and how many of those are "extra" escape characters) have
   * been consumed from a JSON string token while splitting it into individual source lines.
   */
  private record LineSplitOffset(int length, int extraChars) {
    static final LineSplitOffset NONE = new LineSplitOffset(0, 0);

    LineSplitOffset plusLine(int lineLength, int lineExtraChars) {
      // +2 accounts for the JSON-escaped "\n" (backslash + n) separating this line from the next one.
      return new LineSplitOffset(length + lineLength + 2, extraChars + lineExtraChars);
    }

    LineSplitOffset plusLastLine(int lineLength, int lineExtraChars) {
      return new LineSplitOffset(length + lineLength, extraChars + lineExtraChars);
    }

    LineSplitOffset plusNewline() {
      return new LineSplitOffset(length + 2, extraChars);
    }
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
