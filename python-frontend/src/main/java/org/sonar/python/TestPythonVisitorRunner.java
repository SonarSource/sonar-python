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
package org.sonar.python;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.PythonVisitorContext;
import org.sonar.plugins.python.api.caching.CacheContext;
import org.sonar.plugins.python.api.project.configuration.ProjectConfiguration;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.python.caching.CacheContextImpl;
import org.sonar.python.parser.PythonParser;
import org.sonar.python.project.config.ProjectConfigurationBuilder;
import org.sonar.python.project.config.SignatureBasedAwsLambdaHandlersCollector;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.v2.SymbolTableBuilderV2;
import org.sonar.python.semantic.v2.TypeInferenceV2;
import org.sonar.python.semantic.v2.typetable.ProjectLevelTypeTable;
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
    return createContext(file, null, null);
  }

  public static PythonVisitorContext createContext(File file, @Nullable File workingDirectory) {
    return createContext(file, workingDirectory, null);
  }

  public static PythonVisitorContext createContext(File file, @Nullable File workingDirectory, @Nullable ProjectConfiguration projectConfiguration) {
    return createContext(file, workingDirectory, "", ProjectLevelSymbolTable.empty(), CacheContextImpl.dummyCache(), projectConfiguration);
  }

  public static PythonVisitorContext createContext(File file, @Nullable File workingDirectory, String packageName,
    ProjectLevelSymbolTable projectLevelSymbolTable, CacheContext cacheContext) {
    return createContext(file, workingDirectory, packageName, projectLevelSymbolTable, cacheContext, null);
  }

  public static PythonVisitorContext createContext(File file, @Nullable File workingDirectory, String packageName,
    ProjectLevelSymbolTable projectLevelSymbolTable, CacheContext cacheContext, @Nullable ProjectConfiguration projectConfiguration) {
    TestPythonFile pythonFile = new TestPythonFile(file);
    FileInput rootTree = parseFile(pythonFile);

    var typeTable = new ProjectLevelTypeTable(projectLevelSymbolTable);
    var symbolTableBuilderV2 = new SymbolTableBuilderV2(rootTree);
    var symbolTableV2 = symbolTableBuilderV2.build();
    var typeInference = new TypeInferenceV2(typeTable, pythonFile, symbolTableV2, packageName);
    var moduleType = typeInference.inferModuleType(rootTree);

    if (projectConfiguration == null) {
      var projectConfigurationBuilder = new ProjectConfigurationBuilder();
      new SignatureBasedAwsLambdaHandlersCollector().collect(projectConfigurationBuilder, rootTree, packageName);
      projectConfiguration = projectConfigurationBuilder.build();
    }

    return new PythonVisitorContext.Builder(rootTree, pythonFile)
      .workingDirectory(workingDirectory)
      .packageName(packageName)
      .projectLevelSymbolTable(projectLevelSymbolTable)
      .cacheContext(cacheContext)
      .projectConfiguration(projectConfiguration)
      .typeTable(typeTable)
      .moduleType(moduleType)
      .cfgMap(typeInference.getCfgMap())
      .build();
  }

  public static PythonVisitorContext createContext(MockPythonFile file, @Nullable File workingDirectory, String packageName,
                                                   ProjectLevelSymbolTable projectLevelSymbolTable, CacheContext cacheContext) {
    FileInput rootTree = parseFile(file);
    return new PythonVisitorContext.Builder(rootTree, file)
      .workingDirectory(workingDirectory)
      .packageName(packageName)
      .projectLevelSymbolTable(projectLevelSymbolTable)
      .cacheContext(cacheContext)
      .build();
  }

  public static PythonVisitorContext createNotebookContext(File file, Map<Integer, IPythonLocation> locations, String content, String packageName,
    ProjectLevelSymbolTable projectLevelSymbolTable, CacheContext cacheContext) {
    TestPythonFile pythonFile = new TestPythonFile(file);
    FileInput rootTree = parseNotebookFile(locations, content);
    return new PythonVisitorContext.Builder(rootTree, pythonFile)
      .packageName(packageName)
      .projectLevelSymbolTable(projectLevelSymbolTable)
      .cacheContext(cacheContext)
      .build();
  }

  public static ProjectLevelSymbolTable globalSymbols(List<File> files, File baseDir) {
    return globalSymbols(files, computePackageRoots(files, baseDir));
  }

  public static ProjectLevelSymbolTable globalSymbols(List<File> files, List<String> packageRoots) {
    ProjectLevelSymbolTable projectLevelSymbolTable = ProjectLevelSymbolTable.empty();
    for (File file : files) {
      var pythonFile = new TestPythonFile(file);
      String packageName = pythonPackageName(file, packageRoots);
      fillSymbolTableWithFile(pythonFile, projectLevelSymbolTable, packageName);
    }
    return projectLevelSymbolTable;
  }

  public static ProjectLevelSymbolTable globalSymbols(Map<String, String> pathToContent, String baseDir) {
    File baseDirFile = new File(baseDir);
    List<File> files = pathToContent.keySet().stream().map(path -> new File(baseDir, path)).toList();
    return globalSymbols(pathToContent, baseDir, computePackageRoots(files, baseDirFile));
  }

  public static ProjectLevelSymbolTable globalSymbols(Map<String, String> pathToContent, String baseDir, List<String> packageRoots) {
    ProjectLevelSymbolTable projectLevelSymbolTable = ProjectLevelSymbolTable.empty();
    pathToContent.forEach((path, content) -> {
      var file = new MockPythonFile(baseDir, path, content);
      var packageName = pythonPackageName(file.file(), packageRoots);
      fillSymbolTableWithFile(file, projectLevelSymbolTable, packageName);
    });
    return projectLevelSymbolTable;
  }

  private static void fillSymbolTableWithFile(TestablePythonFile file, ProjectLevelSymbolTable projectLevelSymbolTable, String packageName) {
    if (file.isIPython()) {
      return;
    }
    var astRoot = parseFile(file);
    projectLevelSymbolTable.addModule(astRoot, packageName, file);
  }

  public static FileInput parseNotebookFile(Map<Integer, IPythonLocation> locations, String content) {
    var parser = PythonParser.createIPythonParser();
    var treeMaker = new IPythonTreeMaker(locations);
    var astNode = parser.parse(content);
    return treeMaker.fileInput(astNode);
  }

  private static FileInput parseFile(TestablePythonFile file) {
    var parser = file.isIPython() ? PythonParser.createIPythonParser() : PythonParser.create();
    var treeMaker = file.isIPython() ? new IPythonTreeMaker(Map.of()) : new PythonTreeMaker();

    var astNode = parser.parse(file.content());
    return treeMaker.fileInput(astNode);
  }

  /**
   * Test-only counterpart of the package-root computation in {@code PackageRootResolver}.
   * It lives here because python-frontend cannot depend on python-commons, while downstream
   * test utilities can reuse this class through their existing python-frontend dependency.
   */
  public static List<String> computePackageRoots(List<File> files, File baseDir) {
    String baseDirPath = baseDir.getAbsolutePath();
    List<String> roots = files.stream()
      .map(f -> {
        File current = f.getParentFile();
        while (current != null && !current.getAbsolutePath().equals(baseDirPath)) {
          if (!new File(current, "__init__.py").exists()) {
            break;
          }
          current = current.getParentFile();
        }
        return current != null ? current.getAbsolutePath() : baseDirPath;
      })
      .distinct()
      .sorted(Comparator.comparingInt((String p) -> (int) p.chars().filter(c -> c == File.separatorChar).count()).reversed()
        .thenComparing(Comparator.naturalOrder()))
      .toList();
    return roots.isEmpty() ? List.of(baseDirPath) : roots;
  }

  interface TestablePythonFile extends PythonFile {
    default boolean isIPython() {
      return fileName().endsWith(".ipynb");
    }
  }
  
  public static class MockPythonFile implements TestablePythonFile {

    private final String baseDir;
    private final String path;

    private final String content;

    public MockPythonFile(String baseDir, String path, String content) {
      this.baseDir = baseDir;
      this.path = path;
      this.content = content;
    }

    @Override
    public String content() {
      return content;
    }

    @Override
    public String fileName() {
      var file = new File(path);
      return file.getName();
    }

    @Override
    public URI uri() {
      return new File(baseDir, path).toURI();
    }

    @Override
    public String key() {
      return path;
    }

    public File file() {
      return new File(baseDir, path);
    }

  }
  private static class TestPythonFile implements TestablePythonFile {


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

  }

}
