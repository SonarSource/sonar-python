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
package org.sonar.plugins.python.api;

import com.google.common.annotations.Beta;
import com.sonar.sslr.api.RecognitionException;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.api.SonarProduct;
import org.sonar.plugins.python.api.PythonCheck.PreciseIssue;
import org.sonar.plugins.python.api.caching.CacheContext;
import org.sonar.plugins.python.api.project.configuration.ProjectConfiguration;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.types.v2.ModuleType;
import org.sonar.python.caching.CacheContextImpl;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.SymbolTableBuilder;
import org.sonar.python.semantic.v2.ProjectLevelTypeTable;
import org.sonar.python.semantic.v2.SymbolTableBuilderV2;
import org.sonar.python.semantic.v2.TypeInferenceV2;
import org.sonar.python.semantic.v2.callgraph.CallGraph;
import org.sonar.python.types.v2.TypeChecker;

public class PythonVisitorContext extends PythonInputFileContext {

  private final FileInput rootTree;
  private final RecognitionException parsingException;
  private final TypeChecker typeChecker;
  private ModuleType moduleType = null;
  private final List<PreciseIssue> issues;
  private final ProjectConfiguration projectConfiguration;
  private final CallGraph callGraph;

  private PythonVisitorContext(FileInput rootTree, 
      PythonFile pythonFile, 
      @Nullable File workingDirectory, 
      String packageName, 
      ProjectLevelSymbolTable projectLevelSymbolTable, 
      CacheContext cacheContext,
      SonarProduct sonarProduct,
      ProjectConfiguration projectConfiguration,
      CallGraph callGraph) {

    super(pythonFile, workingDirectory, cacheContext, sonarProduct, projectLevelSymbolTable);
    var symbolTableBuilderV2 = new SymbolTableBuilderV2(rootTree);
    var symbolTable = symbolTableBuilderV2.build();
    buildSymbols(rootTree, pythonFile, packageName, projectLevelSymbolTable);
    var projectLevelTypeTable = new ProjectLevelTypeTable(projectLevelSymbolTable);
    this.moduleType = new TypeInferenceV2(projectLevelTypeTable, pythonFile, symbolTable, packageName).inferModuleType(rootTree);
    this.typeChecker = new TypeChecker(projectLevelTypeTable);
    this.projectConfiguration = projectConfiguration;
    this.callGraph = callGraph;
    this.rootTree = rootTree;
    this.parsingException = null;
    this.issues = new ArrayList<>();
  }
  private static synchronized void buildSymbols(FileInput rootTree, PythonFile pythonFile, String packageName, ProjectLevelSymbolTable projectLevelSymbolTable) {
    var symbolTableBuilder = new SymbolTableBuilder(packageName, pythonFile, projectLevelSymbolTable);
    symbolTableBuilder.visitFileInput(rootTree);
  }

  public PythonVisitorContext(PythonFile pythonFile, RecognitionException parsingException, SonarProduct sonarProduct) {
    super(pythonFile, null, CacheContextImpl.dummyCache(), sonarProduct, ProjectLevelSymbolTable.empty());
    this.rootTree = null;
    this.parsingException = parsingException;
    this.typeChecker = new TypeChecker(new ProjectLevelTypeTable(ProjectLevelSymbolTable.empty()));
    this.projectConfiguration = new ProjectConfiguration();
    this.callGraph = CallGraph.EMPTY;
    this.issues = new ArrayList<>();
  }

  public FileInput rootTree() {
    return rootTree;
  }

  public TypeChecker typeChecker() {
    return typeChecker;
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

  public static class Builder {
    private final PythonFile pythonFile;
    private final FileInput rootTree;

    private Optional<ProjectLevelSymbolTable> projectLevelSymbolTable = Optional.empty();
    private Optional<CacheContext> cacheContext = Optional.empty();
    private Optional<SonarProduct> sonarProduct = Optional.empty();
    private Optional<File> workingDirectory = Optional.empty();
    private Optional<ProjectConfiguration> projectConfiguration = Optional.empty();
    private Optional<CallGraph> callGraph = Optional.empty();
    private Optional<String> packageName = Optional.empty();

    public Builder(FileInput rootTree, PythonFile pythonFile) {
      this.rootTree = rootTree;
      this.pythonFile = pythonFile;
    }

    public Builder workingDirectory(@Nullable File workingDirectory) {
      this.workingDirectory = Optional.ofNullable(workingDirectory);
      return this;
    }
    
    public Builder packageName(String packageName) {
      this.packageName = Optional.ofNullable(packageName);
      return this;
    }

    public Builder projectLevelSymbolTable(ProjectLevelSymbolTable projectLevelSymbolTable) {
      this.projectLevelSymbolTable = Optional.ofNullable(projectLevelSymbolTable);
      return this;
    }

    public Builder cacheContext(CacheContext cacheContext) {
      this.cacheContext = Optional.ofNullable(cacheContext);
      return this;
    }

    public Builder sonarProduct(SonarProduct sonarProduct) {
      this.sonarProduct = Optional.ofNullable(sonarProduct);
      return this;
    }

    public Builder projectConfiguration(ProjectConfiguration projectConfiguration) {
      this.projectConfiguration = Optional.ofNullable(projectConfiguration);
      return this;
    }

    public Builder callGraph(CallGraph callGraph) {
      this.callGraph = Optional.ofNullable(callGraph);
      return this;
    }

    public PythonVisitorContext build() {
      return new PythonVisitorContext(
        rootTree,
        pythonFile,
        workingDirectory.orElse(null),
        packageName.orElse(""),
        projectLevelSymbolTable.orElseGet(ProjectLevelSymbolTable::empty),
        cacheContext.orElseGet(CacheContextImpl::dummyCache),
        sonarProduct.orElse(SonarProduct.SONARQUBE),
        projectConfiguration.orElse(new ProjectConfiguration()),
        callGraph.orElse(CallGraph.EMPTY)
      );
    } 
  }
}
