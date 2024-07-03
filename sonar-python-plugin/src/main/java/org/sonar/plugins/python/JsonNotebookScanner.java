package org.sonar.plugins.python;

import com.fasterxml.jackson.core.JsonLocation;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.sonar.sslr.api.AstNode;
import java.io.IOException;
import java.io.InputStream;
import java.util.LinkedHashMap;
import java.util.Map;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.python.parser.PythonParser;
import org.sonar.python.tree.IPythonTreeMaker;
import org.sonar.python.tree.PythonTreeMaker;

public class JsonNotebookScanner extends Scanner {

  public static class JsonCellLocation {
    private final int line;
    private final int startColumn;
    private final int endColumn;

    public JsonCellLocation(int line, int startColumn, int endColumn) {
      this.line = line;
      this.startColumn = startColumn;
      this.endColumn = endColumn;
    }

    @Override
    public String toString() {
      return "JsonCellLocation{" +
        "line=" + line +
        ", startColumn=" + startColumn +
        ", endColumn=" + endColumn +
        '}';
    }
  }

  protected JsonNotebookScanner(SensorContext context) {
    super(context);
  }

  @Override
  protected String name() {
    return "JSON notebooks";
  }

  public record Cell(int id, String source) {
  }

  @Override
  protected void scanFile(InputFile file) throws IOException {

    var cellToLocation = parseJson(file);

    GeneratedFile generatedFile = new GeneratedFile(file, cellToLocation);

    var parser = PythonParser.createIPythonParser();
    AstNode astNode = parser.parse(generatedFile.content());
    PythonTreeMaker treeMaker = new IPythonTreeMaker();
    FileInput parse = treeMaker.fileInput(astNode);


    System.out.println("aa");

  }

  private static Map<Cell, JsonCellLocation> parseJson(InputFile file) throws IOException {
    InputStream is = file.inputStream();
    // Create and configure an ObjectMapper instance
    ObjectMapper mapper = new ObjectMapper();
    mapper.disable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES);

    Map<Cell, JsonCellLocation> cellLocationMap = new LinkedHashMap<>();
    var cellId = 0;

    // Create a JsonParser instance
    try (JsonParser parser = mapper.getFactory().createParser(is)) {
      while (!parser.isClosed()) {
        JsonToken jsonToken = parser.nextToken();
        if (JsonToken.FIELD_NAME.equals(jsonToken)) {
          String fieldName = parser.currentName();
          if ("cell_type".equals(fieldName)) {
            parser.nextToken();
            if ("code".equals(parser.getValueAsString())) {
              var cell = processCodeCell(parser);
              if (cell != null) {
                cellLocationMap.put(new Cell(cellId++, cell.source), cell.location);
              }
            }
          }
        }
      }

      return cellLocationMap;
    }
  }

  record StringWithLocation(String source, JsonCellLocation location) {

  }

  private static StringWithLocation processCodeCell(JsonParser parser) throws IOException {
    while (!parser.isClosed()) {
      JsonToken jsonToken = parser.nextToken();
      if (JsonToken.FIELD_NAME.equals(jsonToken) && "source".equals(parser.currentName())) {
        jsonToken = parser.nextToken();
        if (jsonToken == JsonToken.START_ARRAY) {
          String cellSource = "";
          JsonLocation tokenLocation = parser.currentTokenLocation();
          while (parser.nextToken() != JsonToken.END_ARRAY) {
            String sourceLine = parser.getValueAsString();
            cellSource += sourceLine + "\n";

          }
          return new StringWithLocation(cellSource,
            new JsonCellLocation(tokenLocation.getLineNr(),
              tokenLocation.getColumnNr(),
              tokenLocation.getColumnNr() + cellSource.length()));
        }
      }
    }
    return null;
  }

  @Override
  protected void processException(Exception e, InputFile file) {

  }
}
