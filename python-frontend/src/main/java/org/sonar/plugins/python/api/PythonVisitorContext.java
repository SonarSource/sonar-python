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
package org.sonar.plugins.python.api;

import com.google.common.annotations.Beta;
import com.sonar.sslr.api.RecognitionException;
import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.api.SonarProduct;
import org.sonar.plugins.python.TestFileClassifier;
import org.sonar.plugins.python.api.PythonCheck.PreciseIssue;
import org.sonar.plugins.python.api.caching.CacheContext;
import org.sonar.plugins.python.api.cfg.ControlFlowGraph;
import org.sonar.plugins.python.api.project.configuration.ProjectConfiguration;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.ModuleType;
import org.sonar.python.caching.CacheContextImpl;
import org.sonar.python.cfg.fixpoint.LiveVariablesAnalysis;
import org.sonar.python.cfg.fixpoint.ReachingDefinitionsAnalysis;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.SymbolTableBuilder;
import org.sonar.python.semantic.v2.SymbolTableBuilderV2;
import org.sonar.python.semantic.v2.TypeInferenceV2;
import org.sonar.python.semantic.v2.callgraph.CallGraph;
import org.sonar.python.semantic.v2.callgraph.CallGraphCollector;
import org.sonar.python.semantic.v2.typetable.ProjectLevelTypeTable;
import org.sonar.python.semantic.v2.typetable.TypeTable;
import org.sonar.python.types.v2.TypeChecker;

public class PythonVisitorContext extends PythonInputFileContext {

  private final FileInput rootTree;
  private final RecognitionException parsingException;
  private final TypeChecker typeChecker;
  private final ModuleType moduleType;
  private final List<PreciseIssue> issues;
  private final ProjectConfiguration projectConfiguration;
  private final CallGraph callGraph;
  private final Map<Tree, ControlFlowGraph> cfgMap;
  private final Map<Tree, LiveVariablesAnalysis>  lvaMap;
  private final ReachingDefinitionsAnalysis reachingDefinitionsAnalysis;
  private final TypeTable typeTable;
  private final boolean likelyTestFile;



  private PythonVisitorContext(FileInput rootTree, 
      PythonFile pythonFile, 
      @Nullable File workingDirectory,
      ProjectLevelSymbolTable projectLevelSymbolTable, 
      CacheContext cacheContext,
      SonarProduct sonarProduct,
      ProjectConfiguration projectConfiguration,
      ModuleType moduleType,
      CallGraph callGraph,
      TypeTable typeTable,
      Map<Tree, ControlFlowGraph> cfgMap,
      @Nullable String testFilePath
    ) {
    super(pythonFile, workingDirectory, cacheContext, sonarProduct, projectLevelSymbolTable);
    this.moduleType = moduleType;
    this.projectConfiguration = projectConfiguration;
    this.callGraph = callGraph;
    this.cfgMap = cfgMap;
    this.reachingDefinitionsAnalysis = new ReachingDefinitionsAnalysis(pythonFile);
    this.lvaMap = new HashMap<>();
    this.rootTree = rootTree;
    this.parsingException = null;
    this.typeTable = typeTable;
    this.typeChecker = new TypeChecker(typeTable);
    this.issues = new ArrayList<>();
    this.likelyTestFile = testFilePath == null ?
      TestFileClassifier.looksLikeTestFile(pythonFile, rootTree) :
      TestFileClassifier.looksLikeTestFile(testFilePath, rootTree);
  }

  public PythonVisitorContext(PythonFile pythonFile, RecognitionException parsingException, SonarProduct sonarProduct) {
    this(pythonFile, parsingException, sonarProduct, null);
  }

  /**
   * Creates a context for a Python file that could not be parsed.
   * @param pythonFile analyzed Python file
   * @param parsingException parsing failure
   * @param sonarProduct analysis product
   * @param testFilePath project-relative path used for test classification
   */
  public PythonVisitorContext(PythonFile pythonFile, RecognitionException parsingException, SonarProduct sonarProduct, @Nullable String testFilePath) {
    super(pythonFile, null, CacheContextImpl.dummyCache(), sonarProduct, ProjectLevelSymbolTable.empty());
    this.rootTree = null;
    this.parsingException = parsingException;
    this.typeTable = new ProjectLevelTypeTable(ProjectLevelSymbolTable.empty());
    this.typeChecker = new TypeChecker(this.typeTable);
    this.projectConfiguration = new ProjectConfiguration();
    this.callGraph = CallGraph.EMPTY;
    this.cfgMap = Map.of();
    this.reachingDefinitionsAnalysis = new ReachingDefinitionsAnalysis(pythonFile);
    this.lvaMap = new HashMap<>();
    this.issues = new ArrayList<>();
    this.moduleType = null;
    this.likelyTestFile = testFilePath == null ?
      TestFileClassifier.looksLikeTestFile(pythonFile, null) :
      TestFileClassifier.looksLikeTestFile(testFilePath, null);
  }

  public FileInput rootTree() {
    return rootTree;
  }

  /**
   * Reports whether this file is likely to contain test code.
   * @return whether the file is likely test content
   */
  public boolean isLikelyTestFile() {
    return likelyTestFile;
  }

  public TypeChecker typeChecker() {
    return typeChecker;
  }

  public TypeTable typeTable() {
    return typeTable;
  }

  public RecognitionException parsingException() {
    return parsingException;
  }

  public void addIssue(PreciseIssue issue) {
    issues.add(issue);
  }

  public List<PreciseIssue> getIssues() {
    return issues;
  }

  public Set<Expression> valuesAtLocation(Name name) {
    return reachingDefinitionsAnalysis.valuesAtLocation(name);
  }

  @CheckForNull
  @Beta
  public ModuleType moduleType() {
    return moduleType;
  }

  public ProjectConfiguration projectConfiguration() {
    return projectConfiguration;
  }

  public CallGraph callGraph() {
    return callGraph;
  }

  @CheckForNull
  public ControlFlowGraph cfg(Tree tree) {
    return cfgMap.get(tree);
  }

  @CheckForNull
  public LiveVariablesAnalysis lva(Tree tree){
    ControlFlowGraph cfg = cfg(tree);
    if (cfg == null) {
      return null;
    }
    return lvaMap.computeIfAbsent(tree, t -> LiveVariablesAnalysis.analyze(cfg));
  }

  public Optional<DjangoViewInfo> getDjangoViewInfo(String fqn) {
    return projectLevelSymbolTable().getDjangoViewInfo(fqn);
  }

  public static class Builder {
    private final PythonFile pythonFile;
    private final FileInput rootTree;

    private Optional<ProjectLevelSymbolTable> projectLevelSymbolTable = Optional.empty();
    private Optional<TypeTable> typeTable = Optional.empty();
    private Optional<CacheContext> cacheContext = Optional.empty();
    private Optional<SonarProduct> sonarProduct = Optional.empty();
    private Optional<File> workingDirectory = Optional.empty();
    private Optional<ProjectConfiguration> projectConfiguration = Optional.empty();
    private Optional<CallGraph> callGraph = Optional.empty();
    private Optional<String> packageName = Optional.empty();
    private Optional<ModuleType> moduleType = Optional.empty();
    private Optional<Map<Tree, ControlFlowGraph>> cfgMap = Optional.empty();
    private Optional<String> testFilePath = Optional.empty();

    public Builder(FileInput rootTree, PythonFile pythonFile) {
      this.rootTree = rootTree;
      this.pythonFile = pythonFile;
    }

    public Builder workingDirectory(@Nullable File workingDirectory) {
      this.workingDirectory = Optional.ofNullable(workingDirectory);
      return this;
    }
    
    public Builder packageName(String packageName) {
      this.packageName = Optional.of(packageName);
      return this;
    }

    public Builder projectLevelSymbolTable(ProjectLevelSymbolTable projectLevelSymbolTable) {
      this.projectLevelSymbolTable = Optional.of(projectLevelSymbolTable);
      return this;
    }
    
    public Builder typeTable(TypeTable typeTable) {
      this.typeTable = Optional.of(typeTable);
      return this;
    }

    public Builder cacheContext(CacheContext cacheContext) {
      this.cacheContext = Optional.of(cacheContext);
      return this;
    }

    public Builder sonarProduct(SonarProduct sonarProduct) {
      this.sonarProduct = Optional.of(sonarProduct);
      return this;
    }

    public Builder projectConfiguration(ProjectConfiguration projectConfiguration) {
      this.projectConfiguration = Optional.of(projectConfiguration);
      return this;
    }

    public Builder moduleType(ModuleType moduleType) {
      this.moduleType = Optional.of(moduleType);
      return this;
    }

    public Builder callGraph(CallGraph callGraph) {
      this.callGraph = Optional.of(callGraph);
      return this;
    }

    // Allows passing pre-computed CFGs when moduleType is set externally and the builder skips type inference.
    public Builder cfgMap(Map<Tree, ControlFlowGraph> cfgMap) {
      this.cfgMap = Optional.of(cfgMap);
      return this;
    }

    /**
     * Sets the project-relative path used for test classification.
     * @param testFilePath project-relative file path
     * @return this builder
     */
    public Builder testFilePath(String testFilePath) {
      this.testFilePath = Optional.of(testFilePath);
      return this;
    }

    public PythonVisitorContext build() {
      var symbolTable = projectLevelSymbolTable.orElseGet(ProjectLevelSymbolTable::empty);
      var pkgName = packageName.orElse("");
      buildSymbols(rootTree, pythonFile, pkgName, symbolTable);
      var finalTypeTable = this.typeTable.orElseGet(() -> new ProjectLevelTypeTable(symbolTable));

      ModuleType mt;
      Map<Tree, ControlFlowGraph> finalCfgMap;
      if (moduleType.isPresent()) {
        mt = moduleType.get();
        finalCfgMap = cfgMap.orElse(Map.of());
      } else {
        var symbolTableBuilderV2 = new SymbolTableBuilderV2(rootTree);
        var symbolTableV2 = symbolTableBuilderV2.build();
        var typeInference = new TypeInferenceV2(finalTypeTable, pythonFile, symbolTableV2, pkgName);
        mt = typeInference.inferModuleType(rootTree);
        finalCfgMap = typeInference.getCfgMap();
      }

      var finalCallGraph = callGraph.orElseGet(() -> CallGraphCollector.collectCallGraph(rootTree));

      return new PythonVisitorContext(
        rootTree,
        pythonFile,
        workingDirectory.orElse(null),
        symbolTable,
        cacheContext.orElseGet(CacheContextImpl::dummyCache),
        sonarProduct.orElse(SonarProduct.SONARQUBE),
        projectConfiguration.orElse(new ProjectConfiguration()),
        mt,
        finalCallGraph,
        finalTypeTable,
        finalCfgMap,
        testFilePath.orElse(null)
      );
    }

    private static synchronized void buildSymbols(FileInput rootTree, PythonFile pythonFile, String packageName, ProjectLevelSymbolTable projectLevelSymbolTable) {
      var symbolTableBuilder = new SymbolTableBuilder(packageName, pythonFile, projectLevelSymbolTable);
      symbolTableBuilder.visitFileInput(rootTree);
    }
  }
}
