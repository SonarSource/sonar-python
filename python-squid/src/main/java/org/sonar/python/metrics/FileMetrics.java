/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2018 SonarSource SA
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
package org.sonar.python.metrics;

import com.sonar.sslr.api.AstNode;
import java.util.ArrayList;
import java.util.List;
import org.sonar.python.PythonVisitorContext;
import org.sonar.python.api.PythonGrammar;

public class FileMetrics {

  private int numberOfStatements;
  private int numberOfClasses;
  private final ComplexityVisitor complexityVisitor = new ComplexityVisitor();
  private final FileLinesVisitor fileLinesVisitor;
  private List<Integer> functionComplexities = new ArrayList<>();

  public FileMetrics(PythonVisitorContext context, boolean ignoreHeaderComments) {
    AstNode rootTree = context.rootTree();
    numberOfStatements = rootTree.getDescendants(PythonGrammar.STATEMENT).size();
    numberOfClasses = rootTree.getDescendants(PythonGrammar.CLASSDEF).size();
    complexityVisitor.scanFile(context);
    fileLinesVisitor = new FileLinesVisitor(ignoreHeaderComments);
    fileLinesVisitor.scanFile(context);
    for (AstNode functionDef : rootTree.getDescendants(PythonGrammar.FUNCDEF)) {
      functionComplexities.add(ComplexityVisitor.complexity(functionDef));
    }
  }

  public int numberOfStatements() {
    return numberOfStatements;
  }

  public int numberOfFunctions() {
    return functionComplexities.size();
  }

  public int numberOfClasses() {
    return numberOfClasses;
  }

  public int complexity() {
    return complexityVisitor.getComplexity();
  }

  public List<Integer> functionComplexities() {
    return functionComplexities;
  }

  public FileLinesVisitor fileLinesVisitor() {
    return fileLinesVisitor;
  }

}
