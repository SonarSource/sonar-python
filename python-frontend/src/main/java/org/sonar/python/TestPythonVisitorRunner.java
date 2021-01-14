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
package org.sonar.python;

import com.sonar.sslr.api.AstNode;
import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.List;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.PythonVisitorContext;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.python.parser.PythonParser;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.tree.PythonTreeMaker;

import static org.sonar.python.semantic.SymbolUtils.pythonPackageName;

public class TestPythonVisitorRunner {

  private TestPythonVisitorRunner() {
  }

  public static PythonVisitorContext scanFile(File file, PythonCheck... visitors) {
    PythonVisitorContext context = createContext(file);
    for (PythonCheck visitor : visitors) {
      visitor.scanFile(context);
    }
    return context;
  }

  public static PythonVisitorContext createContext(File file) {
    return createContext(file, null);
  }

  public static PythonVisitorContext createContext(File file, @Nullable File workingDirectory) {
    return createContext(file, workingDirectory, "", ProjectLevelSymbolTable.empty());
  }

  public static PythonVisitorContext createContext(File file, @Nullable File workingDirectory, String packageName, ProjectLevelSymbolTable projectLevelSymbolTable) {
    PythonParser parser = PythonParser.create();
    TestPythonFile pythonFile = new TestPythonFile(file);
    AstNode astNode = parser.parse(pythonFile.content());
    FileInput rootTree = new PythonTreeMaker().fileInput(astNode);
    return new PythonVisitorContext(rootTree, pythonFile, workingDirectory, packageName, projectLevelSymbolTable);
  }

  public static ProjectLevelSymbolTable globalSymbols(List<File> files, File baseDir) {
    ProjectLevelSymbolTable projectLevelSymbolTable = new ProjectLevelSymbolTable();
    for (File file : files) {
      TestPythonFile pythonFile = new TestPythonFile(file);
      AstNode astNode = PythonParser.create().parse(pythonFile.content());
      FileInput astRoot = new PythonTreeMaker().fileInput(astNode);
      String packageName = pythonPackageName(file, baseDir);
      projectLevelSymbolTable.addModule(astRoot, packageName, pythonFile);
    }
    return projectLevelSymbolTable;
  }

  private static class TestPythonFile implements PythonFile {

    private final File file;

    public TestPythonFile(File file) {
      this.file = file;
    }

    @Override
    public String content() {
      try {
        return new String(Files.readAllBytes(file.toPath()), StandardCharsets.UTF_8);
      } catch (IOException e) {
        throw new IllegalStateException("Cannot read " + file, e);
      }
    }

    @Override
    public String fileName() {
      return file.getName();
    }

    @Override
    public URI uri() {
      return file.toURI();
    }

  }

}
