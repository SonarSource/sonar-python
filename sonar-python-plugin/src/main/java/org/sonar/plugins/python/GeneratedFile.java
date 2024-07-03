package org.sonar.plugins.python;

import java.net.URI;
import java.util.Map;
import java.util.stream.Collectors;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.plugins.python.api.PythonFile;

public class GeneratedFile implements PythonFile {

  InputFile inputFile;

  String contents;

  Map<JsonNotebookScanner.Cell, JsonNotebookScanner.JsonCellLocation> cellToLocation;

  private static final String SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER = "#SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER\n";

  public GeneratedFile(InputFile inputFile, Map<JsonNotebookScanner.Cell, JsonNotebookScanner.JsonCellLocation> cellToLocation) {
    this.inputFile = inputFile;
    this.cellToLocation = cellToLocation;

     // Join all cells, with the delimiter

    this.contents = cellToLocation.keySet().stream()
      .map(JsonNotebookScanner.Cell::source)
      .collect(Collectors.joining(SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER));

    System.out.println("aa");
  }

  @Override
  public String content() {
    return contents;
  }

  @Override
  public String fileName() {
    return inputFile.filename();
  }

  @Override
  public URI uri() {
    return inputFile.uri();
  }

  @Override
  public String key() {
    return inputFile.key();
  }
}
