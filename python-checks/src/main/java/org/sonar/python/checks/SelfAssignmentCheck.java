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
package org.sonar.python.checks;

import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.AstNodeType;
import java.util.HashSet;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.python.PythonBuiltinFunctions;
import org.sonar.python.PythonCheck;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.api.PythonKeyword;
import org.sonar.python.api.PythonPunctuator;
import org.sonar.sslr.ast.AstSelect;

@Rule(key = SelfAssignmentCheck.CHECK_KEY)
public class SelfAssignmentCheck extends PythonCheck {

  public static final String CHECK_KEY = "S1656";

  public static final String MESSAGE = "Remove or correct this useless self-assignment.";

  private Set<String> importedNames = new HashSet<>();

  @Override
  public void visitFile(AstNode node) {
    importedNames.clear();
  }

  @Override
  public Set<AstNodeType> subscribedKinds() {
    return immutableSet(
      PythonGrammar.EXPRESSION_STMT,
      PythonGrammar.IMPORT_NAME,
      PythonGrammar.IMPORT_AS_NAME);
  }

  @Override
  public void visitNode(AstNode node) {
    if (node.is(PythonGrammar.IMPORT_NAME)) {
      for (AstNode dottedAsName : node.select().children(PythonGrammar.DOTTED_AS_NAMES).children(PythonGrammar.DOTTED_AS_NAME)) {
        AstNode importedName = dottedAsName.getFirstChild().getLastChild(PythonGrammar.NAME);
        addImportedName(dottedAsName, importedName);
      }

    } else if (node.is(PythonGrammar.IMPORT_AS_NAME)) {
      AstNode importedName = node.getFirstChild(PythonGrammar.NAME);
      addImportedName(node, importedName);

    } else {
      checkExpressionStatement(node);
    }
  }

  private void checkExpressionStatement(AstNode node) {
    for (AstNode assignOperator : node.getChildren(PythonPunctuator.ASSIGN, PythonGrammar.ANNASSIGN)) {
      AstNode assigned = assignOperator.getPreviousSibling();
      if (assignOperator.is(PythonGrammar.ANNASSIGN)) {
        assignOperator = assignOperator.getFirstChild(PythonPunctuator.ASSIGN);
        if (assigned.is(PythonGrammar.TESTLIST_STAR_EXPR) && assigned.getNumberOfChildren() == 1) {
          assigned =  assigned.getFirstChild();
        }
      }
      if (assignOperator != null && CheckUtils.equalNodes(assigned, assignOperator.getNextSibling()) && !isException(node, assigned)) {
        addIssue(assignOperator, MESSAGE);
      }
    }
  }

  private void addImportedName(AstNode node, AstNode importedName) {
    AstNode name = importedName;
    AstNode as = node.getFirstChild(PythonKeyword.AS);
    if (as != null) {
      name = as.getNextSibling();
    }
    importedNames.add(name.getTokenValue());
  }

  private boolean isException(AstNode expressionStatement, AstNode assigned) {
    AstSelect potentialFunctionCalls = assigned.select()
      .descendants(PythonGrammar.TRAILER)
      .children(PythonPunctuator.LPARENTHESIS);
    if (!potentialFunctionCalls.isEmpty()) {
      return true;
    }

    if (assigned.getTokens().size() == 1) {
      String tokenValue = assigned.getTokenValue();
      if (importedNames.contains(tokenValue) || PythonBuiltinFunctions.contains(tokenValue)) {
        return true;
      }
    }

    AstNode suite = expressionStatement.getFirstAncestor(PythonGrammar.SUITE);
    return suite != null && suite.getParent().is(PythonGrammar.CLASSDEF, PythonGrammar.TRY_STMT);
  }
}
