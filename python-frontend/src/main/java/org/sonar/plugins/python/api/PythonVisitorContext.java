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
import org.sonar.python.types.v2.TypeChecker;

public class PythonVisitorContext extends PythonInputFileContext {

  private final FileInput rootTree;
  private final RecognitionException parsingException;
  private final TypeChecker typeChecker;
  private ModuleType moduleType = null;
  private final List<PreciseIssue> issues;
  private final ProjectConfiguration projectConfiguration;

  public PythonVisitorContext(FileInput rootTree, PythonFile pythonFile, @Nullable File workingDirectory, String packageName) {
    this(rootTree, pythonFile, workingDirectory, packageName, new ProjectConfiguration());
  }

  public PythonVisitorContext(FileInput rootTree, PythonFile pythonFile, @Nullable File workingDirectory, String packageName, ProjectConfiguration projectConfiguration) {
    super(pythonFile, workingDirectory, CacheContextImpl.dummyCache(), ProjectLevelSymbolTable.empty());
    buildSymbols(rootTree, pythonFile, packageName);
    var symbolTable = new SymbolTableBuilderV2(rootTree).build();
    var projectLevelTypeTable = new ProjectLevelTypeTable(ProjectLevelSymbolTable.empty());

    this.rootTree = rootTree;
    this.parsingException = null;
    this.moduleType = new TypeInferenceV2(projectLevelTypeTable, pythonFile, symbolTable, packageName).inferModuleType(rootTree);
    this.typeChecker = new TypeChecker(projectLevelTypeTable);
    this.projectConfiguration = projectConfiguration;
    this.issues = new ArrayList<>();
  }

  public PythonVisitorContext(FileInput rootTree, PythonFile pythonFile, @Nullable File workingDirectory, String packageName,
    ProjectLevelSymbolTable projectLevelSymbolTable, CacheContext cacheContext) {
    this(rootTree, pythonFile, workingDirectory, packageName, projectLevelSymbolTable, cacheContext, new ProjectConfiguration());
  }

  public PythonVisitorContext(FileInput rootTree, PythonFile pythonFile, @Nullable File workingDirectory, String packageName,
    ProjectLevelSymbolTable projectLevelSymbolTable, CacheContext cacheContext, ProjectConfiguration projectConfiguration) {
    super(pythonFile, workingDirectory, cacheContext, projectLevelSymbolTable);

    buildSymbols(rootTree, pythonFile, packageName, projectLevelSymbolTable);
    var symbolTable = new SymbolTableBuilderV2(rootTree).build();
    var projectLevelTypeTable = new ProjectLevelTypeTable(projectLevelSymbolTable);

    this.rootTree = rootTree;
    this.parsingException = null;
    this.moduleType = new TypeInferenceV2(projectLevelTypeTable, pythonFile, symbolTable, packageName).inferModuleType(rootTree);
    this.typeChecker = new TypeChecker(projectLevelTypeTable);
    this.projectConfiguration = projectConfiguration;
    this.issues = new ArrayList<>();
  }

  public PythonVisitorContext(FileInput rootTree, PythonFile pythonFile, @Nullable File workingDirectory, String packageName,
    ProjectLevelSymbolTable projectLevelSymbolTable, CacheContext cacheContext, SonarProduct sonarProduct) {
    super(pythonFile, workingDirectory, cacheContext, sonarProduct, projectLevelSymbolTable);
    var symbolTableBuilderV2 = new SymbolTableBuilderV2(rootTree);
    var symbolTable = symbolTableBuilderV2.build();
    buildSymbols(rootTree, pythonFile, packageName, projectLevelSymbolTable);
    var projectLevelTypeTable = new ProjectLevelTypeTable(projectLevelSymbolTable);
    this.moduleType = new TypeInferenceV2(projectLevelTypeTable, pythonFile, symbolTable, packageName).inferModuleType(rootTree);
    this.typeChecker = new TypeChecker(projectLevelTypeTable);
    this.projectConfiguration = new ProjectConfiguration();
    this.rootTree = rootTree;
    this.parsingException = null;
    this.issues = new ArrayList<>();
  }

  private static synchronized void buildSymbols(FileInput rootTree, PythonFile pythonFile, String packageName) {
    buildSymbols(rootTree, pythonFile, packageName, ProjectLevelSymbolTable.empty());
  }

  private static synchronized void buildSymbols(FileInput rootTree, PythonFile pythonFile, String packageName, ProjectLevelSymbolTable projectLevelSymbolTable) {
    var symbolTableBuilder = new SymbolTableBuilder(packageName, pythonFile, projectLevelSymbolTable);
    symbolTableBuilder.visitFileInput(rootTree);
  }

  public PythonVisitorContext(PythonFile pythonFile, RecognitionException parsingException) {
    super(pythonFile, null, CacheContextImpl.dummyCache(), ProjectLevelSymbolTable.empty());
    this.rootTree = null;
    this.parsingException = parsingException;
    this.typeChecker = new TypeChecker(new ProjectLevelTypeTable(ProjectLevelSymbolTable.empty()));
    this.projectConfiguration = new ProjectConfiguration();
    this.issues = new ArrayList<>();
  }

  public PythonVisitorContext(PythonFile pythonFile, RecognitionException parsingException, SonarProduct sonarProduct) {
    super(pythonFile, null, CacheContextImpl.dummyCache(), sonarProduct, ProjectLevelSymbolTable.empty());
    this.rootTree = null;
    this.parsingException = parsingException;
    this.typeChecker = new TypeChecker(new ProjectLevelTypeTable(ProjectLevelSymbolTable.empty()));
    this.projectConfiguration = new ProjectConfiguration();
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
}
