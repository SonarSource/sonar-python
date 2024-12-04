/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
package org.sonar.python;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.PythonVisitorContext;
import org.sonar.plugins.python.api.caching.CacheContext;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.python.caching.CacheContextImpl;
import org.sonar.python.parser.PythonParser;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.tree.IPythonTreeMaker;
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

  public static PythonVisitorContext scanNotebookFile(File file, Map<Integer, IPythonLocation> locations, String content, PythonCheck... visitors) {
    PythonVisitorContext context = createNotebookContext(file, locations, content, "", ProjectLevelSymbolTable.empty(), CacheContextImpl.dummyCache());
    for (PythonCheck visitor : visitors) {
      visitor.scanFile(context);
    }
    return context;
  }

  public static PythonVisitorContext createContext(File file) {
    return createContext(file, null);
  }

  public static PythonVisitorContext createContext(File file, @Nullable File workingDirectory) {
    return createContext(file, workingDirectory, "", ProjectLevelSymbolTable.empty(), CacheContextImpl.dummyCache());
  }

  public static PythonVisitorContext createContext(File file, @Nullable File workingDirectory, String packageName,
    ProjectLevelSymbolTable projectLevelSymbolTable, CacheContext cacheContext) {
    TestPythonFile pythonFile = new TestPythonFile(file);
    FileInput rootTree = parseFile(pythonFile);
    return new PythonVisitorContext(rootTree, pythonFile, workingDirectory, packageName, projectLevelSymbolTable, cacheContext);
  }

  public static PythonVisitorContext createNotebookContext(File file, Map<Integer, IPythonLocation> locations, String content, String packageName,
    ProjectLevelSymbolTable projectLevelSymbolTable, CacheContext cacheContext) {
    TestPythonFile pythonFile = new TestPythonFile(file);
    FileInput rootTree = parseNotebookFile(locations, content);
    return new PythonVisitorContext(rootTree, pythonFile, null, packageName, projectLevelSymbolTable, cacheContext);
  }

  public static ProjectLevelSymbolTable globalSymbols(List<File> files, File baseDir) {
    ProjectLevelSymbolTable projectLevelSymbolTable = ProjectLevelSymbolTable.empty();
    for (File file : files) {
      var pythonFile = new TestPythonFile(file);
      if (pythonFile.isIPython()) {
        continue;
      }
      var astRoot = parseFile(pythonFile);
      String packageName = pythonPackageName(file, baseDir.getAbsolutePath());
      projectLevelSymbolTable.addModule(astRoot, packageName, pythonFile);
    }
    return projectLevelSymbolTable;
  }

  public static FileInput parseNotebookFile(Map<Integer, IPythonLocation> locations, String content) {
    var parser = PythonParser.createIPythonParser();
    var treeMaker = new IPythonTreeMaker(locations);
    var astNode = parser.parse(content);
    return treeMaker.fileInput(astNode);
  }

  private static FileInput parseFile(TestPythonFile file) {
    var parser = file.isIPython() ? PythonParser.createIPythonParser() : PythonParser.create();
    var treeMaker = file.isIPython() ? new IPythonTreeMaker(Map.of()) : new PythonTreeMaker();

    var astNode = parser.parse(file.content());
    return treeMaker.fileInput(astNode);
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

    @Override
    public String key() {
      return file.getPath();
    }

    public boolean isIPython() {
      return fileName().endsWith(".ipynb");
    }

  }

}
