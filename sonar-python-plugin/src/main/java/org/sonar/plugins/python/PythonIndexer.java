/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
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

import com.sonar.sslr.api.AstNode;
import java.io.IOException;
import java.net.URI;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.api.utils.log.Logger;
import org.sonar.api.utils.log.Loggers;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.python.parser.PythonParser;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.tree.PythonTreeMaker;

import static org.sonar.python.semantic.SymbolUtils.pythonPackageName;

public class PythonIndexer {

  private static final Logger LOG = Loggers.get(PythonIndexer.class);

  private final Map<URI, String> packageNames = new HashMap<>();
  private final PythonParser parser = PythonParser.create();
  private final ProjectLevelSymbolTable projectLevelSymbolTable = new ProjectLevelSymbolTable();


  void buildOnce(SensorContext context, List<InputFile> files) {
    LOG.debug("Input files for indexing: " + files);
    // computes "globalSymbolsByModuleName"
    long startTime = System.currentTimeMillis();
    GlobalSymbolsScanner globalSymbolsStep = new GlobalSymbolsScanner(context);
    globalSymbolsStep.execute(files, context);
    long stopTime = System.currentTimeMillis() - startTime;
    LOG.debug("Time to build the project level symbol table: " + stopTime + "ms");
  }

  String packageName(URI uri) {
    return packageNames.get(uri);
  }

  ProjectLevelSymbolTable projectLevelSymbolTable() {
    return projectLevelSymbolTable;
  }

  private class GlobalSymbolsScanner extends Scanner {

    private GlobalSymbolsScanner(SensorContext context) {
      super(context);
    }

    @Override
    protected String name() {
      return "global symbols computation";
    }

    @Override
    protected void scanFile(InputFile inputFile) throws IOException {
      AstNode astNode = parser.parse(inputFile.contents());
      FileInput astRoot = new PythonTreeMaker().fileInput(astNode);
      String packageName = pythonPackageName(inputFile.file(), context.fileSystem().baseDir());
      packageNames.put(inputFile.uri(), packageName);
      PythonFile pythonFile = SonarQubePythonFile.create(inputFile);
      projectLevelSymbolTable.addModule(astRoot, packageName, pythonFile);
    }

    @Override
    protected void processException(Exception e, InputFile file) {
      LOG.debug("Unable to construct project-level symbol table for file: " + file.toString());
      LOG.debug(e.getMessage());
    }
  }
}
