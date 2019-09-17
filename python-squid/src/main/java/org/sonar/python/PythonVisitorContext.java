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
package org.sonar.python;

import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.RecognitionException;
import java.util.ArrayList;
import java.util.List;
import javax.annotation.Nullable;
import org.sonar.python.PythonCheck.PreciseIssue;
import org.sonar.python.api.tree.PyFileInputTree;
import org.sonar.python.semantic.SymbolTable;
import org.sonar.python.semantic.SymbolTableBuilderVisitor;
import org.sonar.python.tree.PyFileInputTreeImpl;

public class PythonVisitorContext {

  private final PyFileInputTreeImpl rootTree;
  private final PythonFile pythonFile;
  private final RecognitionException parsingException;
  private final AstNode rootAst;
  private SymbolTable symbolTable = null;
  private List<PreciseIssue> issues = new ArrayList<>();


  public PythonVisitorContext(AstNode rootAst, PyFileInputTree rootTree, PythonFile pythonFile) {
    this(rootAst, rootTree, pythonFile, null);
    SymbolTableBuilderVisitor symbolTableBuilderVisitor = new SymbolTableBuilderVisitor();
    symbolTableBuilderVisitor.scanFile(this);
    symbolTable = symbolTableBuilderVisitor.symbolTable();
  }

  public PythonVisitorContext(PythonFile pythonFile, RecognitionException parsingException) {
    this(null, null, pythonFile, parsingException);
  }

  private PythonVisitorContext(@Nullable AstNode rootAst, @Nullable PyFileInputTree rootTree,
                               PythonFile pythonFile, @Nullable RecognitionException parsingException) {
    this.rootAst = rootAst;
    this.rootTree = (PyFileInputTreeImpl) rootTree;
    this.pythonFile = pythonFile;
    this.parsingException = parsingException;
  }

  public PyFileInputTree rootTree() {
    return rootTree;
  }

  public AstNode rootAstNode() {
    return rootAst;
  }

  public PythonFile pythonFile() {
    return pythonFile;
  }

  public RecognitionException parsingException() {
    return parsingException;
  }

  public SymbolTable symbolTable() {
    return symbolTable;
  }

  public void addIssue(PreciseIssue issue) {
    issues.add(issue);
  }

  public List<PreciseIssue> getIssues() {
    return issues;
  }
}
