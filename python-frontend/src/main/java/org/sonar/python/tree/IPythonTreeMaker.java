/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2023 SonarSource SA
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
package org.sonar.python.tree;

import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.GenericTokenType;
import java.util.ArrayList;
import java.util.Collection;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.sonar.plugins.python.api.tree.CellMagicStatement;
import org.sonar.plugins.python.api.tree.DynamicObjectInfoStatement;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.LineMagic;
import org.sonar.plugins.python.api.tree.LineMagicStatement;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.StatementList;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.DocstringExtractor;
import org.sonar.python.api.IPythonGrammar;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.api.PythonPunctuator;

public class IPythonTreeMaker extends PythonTreeMaker {

  @Override
  public FileInput fileInput(AstNode astNode) {
    StatementList statementList = astNode.getChildren(IPythonGrammar.CELL, IPythonGrammar.MAGIC_CELL)
      .stream()
      .flatMap(this::getStatementsFromCell)
      .collect(Collectors.collectingAndThen(Collectors.toList(), l -> l.isEmpty()? null : new StatementListImpl(l)));
    Token endOfFile = toPyToken(astNode.getFirstChild(GenericTokenType.EOF).getToken());
    FileInputImpl pyFileInputTree = new FileInputImpl(statementList, endOfFile, DocstringExtractor.extractDocstring(statementList));
    setParents(pyFileInputTree);
    pyFileInputTree.accept(new ExceptGroupJumpInstructionsCheck());
    return pyFileInputTree;
  }

  private Stream<Statement> getStatementsFromCell(AstNode cell) {
    if (cell.is(IPythonGrammar.CELL)) {
      return getStatements(cell).stream().map(this::statement);
    } else {
      return Stream.of(cell.getFirstChild(IPythonGrammar.CELL_MAGIC_STATEMENT)).map(IPythonTreeMaker::cellMagicStatement);
    }
  }

  @Override
  protected Statement statement(StatementWithSeparator statementWithSeparator) {
    var astNode = statementWithSeparator.statement();

    if (astNode.is(IPythonGrammar.LINE_MAGIC_STATEMENT)) {
      return lineMagicStatement(astNode);
    }
    if (astNode.is(IPythonGrammar.DYNAMIC_OBJECT_INFO_STATEMENT)) {
      return dynamicObjectInfoStatement(astNode);
    }
    return super.statement(statementWithSeparator);
  }

  @Override
  protected Expression annotatedRhs(AstNode annotatedRhs) {
    var child = annotatedRhs.getFirstChild();
    if (child.is(IPythonGrammar.LINE_MAGIC)) {
      return lineMagic(child);
    }
    return super.annotatedRhs(annotatedRhs);
  }

  private static CellMagicStatement cellMagicStatement(AstNode astNode) {
    var tokens = astNode.getChildren()
      .stream()
      .map(AstNode::getTokens)
      .flatMap(Collection::stream)
      .map(IPythonTreeMaker::toPyToken)
      .collect(Collectors.toList());
    return new CellMagicStatementImpl(tokens);
  }

  protected LineMagicStatement lineMagicStatement(AstNode astNode) {
    var lineMagic = lineMagic(astNode.getFirstChild(IPythonGrammar.LINE_MAGIC));
    return new LineMagicStatementImpl(lineMagic);
  }

  protected DynamicObjectInfoStatement dynamicObjectInfoStatement(AstNode astNode) {
    var questionMarksBefore = new ArrayList<Tree>();
    var children = new ArrayList<Tree>();
    var questionMarksAfter = new ArrayList<Tree>();

    var nodeChildren = astNode.getChildren();
    var currentList = questionMarksBefore;
    for (int i = 0; i < nodeChildren.size(); i++) {
      var nodeChild = nodeChildren.get(i);
      if (nodeChild.is(PythonPunctuator.QUESTION_MARK)) {
        if (currentList == children) {
          currentList = questionMarksAfter;
        }
        var nodeChildToken = toPyToken(nodeChild.getToken());
        currentList.add(nodeChildToken);
      } else {
        currentList = children;
        nodeChild.getTokens()
          .stream()
          .map(PythonTreeMaker::toPyToken)
          .forEach(currentList::add);
      }
    }
    return new DynamicObjectInfoStatementImpl(questionMarksBefore, children, questionMarksAfter);
  }

  protected LineMagic lineMagic(AstNode astNode) {
    var percent = toPyToken(astNode.getFirstChild().getToken());
    var name = toPyToken(astNode.getFirstChild(PythonGrammar.NAME).getToken());

    var tokens = astNode.getChildren()
      .stream()
      .skip(2)
      .map(AstNode::getTokens)
      .flatMap(Collection::stream)
      .map(IPythonTreeMaker::toPyToken)
      .collect(Collectors.toList());
    return new LineMagicImpl(percent, name, tokens);
  }

}
