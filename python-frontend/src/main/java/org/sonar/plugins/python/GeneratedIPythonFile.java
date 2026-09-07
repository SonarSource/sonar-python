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

import java.io.IOException;
import java.util.Map;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.python.IPythonLocation;
import org.sonar.python.parser.IPythonParserConfiguration;


public class GeneratedIPythonFile implements PythonInputFile {

  InputFile wrappedFile;

  private String pythonContent;

  private Map<Integer, IPythonLocation> locationMap;

  private final NotebookDialect dialect;

  private final IPythonParserConfiguration parserConfiguration;

  public GeneratedIPythonFile(InputFile wrappedFile, String pythonContent, Map<Integer, IPythonLocation> locationMap) {
    this(wrappedFile, pythonContent, locationMap, NotebookDialect.IPYTHON, IPythonParserConfiguration.empty());
  }

  public GeneratedIPythonFile(InputFile wrappedFile, String pythonContent, Map<Integer, IPythonLocation> locationMap, NotebookDialect dialect,
    IPythonParserConfiguration parserConfiguration) {
    this.locationMap = locationMap;
    this.wrappedFile = wrappedFile;
    this.pythonContent = pythonContent;
    this.dialect = dialect;
    this.parserConfiguration = parserConfiguration;
  }

  public Map<Integer, IPythonLocation> locationMap() {
    return locationMap;
  }

  public NotebookDialect dialect() {
    return dialect;
  }

  public IPythonParserConfiguration parserConfiguration() {
    return parserConfiguration;
  }

  @Override
  public InputFile wrappedFile() {
    return wrappedFile;
  }

  @Override
  public Kind kind() {
    return Kind.IPYTHON;
  }

  @Override
  public String toString() {
    return wrappedFile.toString();
  }

  @Override
  public String contents() throws IOException {
    return pythonContent;
  }

}
