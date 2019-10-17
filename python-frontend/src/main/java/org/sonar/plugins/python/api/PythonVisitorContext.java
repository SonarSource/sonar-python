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
package org.sonar.plugins.python.api;

import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.RecognitionException;
import java.util.ArrayList;
import java.util.List;
import org.sonar.plugins.python.api.PythonCheck.PreciseIssue;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.python.PythonFile;
import org.sonar.python.semantic.SymbolTableBuilder;

public class PythonVisitorContext {

  private final FileInput rootTree;
  private final PythonFile pythonFile;
  private final RecognitionException parsingException;
  private final AstNode rootAst;
  private List<PreciseIssue> issues = new ArrayList<>();


  public PythonVisitorContext(AstNode rootAst, FileInput rootTree, PythonFile pythonFile) {
    this.rootAst = rootAst;
    this.rootTree = rootTree;
    this.pythonFile = pythonFile;
    this.parsingException = null;
    new SymbolTableBuilder().visitFileInput(rootTree);
  }

  public PythonVisitorContext(PythonFile pythonFile, RecognitionException parsingException) {
    this.rootAst = null;
    this.rootTree = null;
    this.pythonFile = pythonFile;
    this.parsingException = parsingException;
  }

  public FileInput rootTree() {
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

  public void addIssue(PreciseIssue issue) {
    issues.add(issue);
  }

  public List<PreciseIssue> getIssues() {
    return issues;
  }
}
