/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2019 SonarSource SA
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

import com.intellij.psi.util.PsiTreeUtil;
import com.jetbrains.python.psi.PyClass;
import com.jetbrains.python.psi.PyFile;
import com.jetbrains.python.psi.PyFunction;
import com.jetbrains.python.psi.PyStatement;
import java.util.ArrayList;
import java.util.List;
import org.sonar.python.PythonVisitorContext;

public class FileMetrics {

  private int numberOfStatements;
  private int numberOfClasses;
  private int cyclomaticComplexity;
  private final CognitiveComplexityVisitor cognitiveComplexityVisitor = new CognitiveComplexityVisitor(null);
  private final MetricsVisitor metricsVisitor;
  private List<Integer> functionComplexities = new ArrayList<>();

  public FileMetrics(PythonVisitorContext context, boolean ignoreHeaderComments, PyFile pyFile) {
    numberOfStatements = PsiTreeUtil.findChildrenOfType(pyFile, PyStatement.class).size();
    numberOfClasses = PsiTreeUtil.findChildrenOfType(pyFile, PyClass.class).size();
    cyclomaticComplexity = ComplexityVisitor.complexity(pyFile);
    cognitiveComplexityVisitor.scanFile(context);
    metricsVisitor = new MetricsVisitor(ignoreHeaderComments);
    pyFile.accept(metricsVisitor);
    for (PyFunction functionDef : PsiTreeUtil.findChildrenOfType(pyFile, PyFunction.class)) {
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
    return cyclomaticComplexity;
  }

  public int cognitiveComplexity() {
    return cognitiveComplexityVisitor.getComplexity();
  }

  public List<Integer> functionComplexities() {
    return functionComplexities;
  }

  public MetricsVisitor metricsVisitor() {
    return metricsVisitor;
  }

}
