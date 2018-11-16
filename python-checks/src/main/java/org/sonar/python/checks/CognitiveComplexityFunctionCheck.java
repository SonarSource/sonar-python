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
import java.util.List;
import java.util.Set;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.check.RuleProperty;
import org.sonar.python.IssueLocation;
import org.sonar.python.PythonCheck;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.api.PythonKeyword;
import org.sonar.python.api.PythonPunctuator;

@Rule(key = CognitiveComplexityFunctionCheck.CHECK_KEY)
public class CognitiveComplexityFunctionCheck extends PythonCheck {

  private static final String MESSAGE = "Refactor this function to reduce its Cognitive Complexity from %s to the %s allowed.";
  public static final String CHECK_KEY = "S3776";
  private static final int DEFAULT_THRESHOLD = 15;

  private AstNode currentFunction = null;
  private int complexity;
  private int nestingLevel;
  private Set<IssueLocation> secondaryLocations = new HashSet<>();

  @RuleProperty(
    key = "threshold",
    description = "The maximum authorized complexity.",
    defaultValue = "" + DEFAULT_THRESHOLD)
  private int threshold = DEFAULT_THRESHOLD;

  public void setThreshold(int threshold) {
    this.threshold = threshold;
  }

  @Override
  public Set<AstNodeType> subscribedKinds() {
    return immutableSet(
      PythonGrammar.IF_STMT,
      PythonKeyword.ELIF,
      PythonKeyword.ELSE,

      PythonGrammar.WHILE_STMT,
      PythonGrammar.FOR_STMT,
      PythonGrammar.EXCEPT_CLAUSE,

      PythonGrammar.AND_TEST,
      PythonGrammar.OR_TEST,

      PythonGrammar.TEST,

      PythonGrammar.FUNCDEF,
      PythonGrammar.SUITE);
  }

  @Override
  public void visitNode(AstNode astNode) {
    if (astNode.is(PythonGrammar.FUNCDEF) && currentFunction == null) {
      currentFunction = astNode;
      complexity = 0;
      nestingLevel = 0;
      secondaryLocations.clear();
    }

    if (currentFunction != null) {
      if (astNode.is(PythonGrammar.SUITE) && incrementsNestingLevel(astNode)) {
        nestingLevel++;
      }

      checkComplexity(astNode);
    }
  }

  private void checkComplexity(AstNode astNode) {
    if (astNode.is(PythonGrammar.IF_STMT, PythonGrammar.WHILE_STMT, PythonGrammar.FOR_STMT, PythonGrammar.EXCEPT_CLAUSE)) {
      incrementWithNesting(astNode.getFirstChild());
    }

    if (astNode.is(PythonKeyword.ELIF) || (astNode.is(PythonKeyword.ELSE) && astNode.getNextSibling().is(PythonPunctuator.COLON))) {
      incrementWithoutNesting(astNode);
    }

    if (astNode.is(PythonGrammar.AND_TEST, PythonGrammar.OR_TEST)) {
      incrementWithoutNesting(astNode.getFirstChild(PythonKeyword.AND, PythonKeyword.OR));
    }

    // conditional expression
    if (astNode.is(PythonGrammar.TEST) && astNode.hasDirectChildren(PythonKeyword.IF)) {
      incrementWithNesting(astNode.getFirstChild(PythonKeyword.IF));
    }
  }

  @Override
  public void leaveNode(AstNode astNode) {
    if (currentFunction == null) {
      return;
    }

    if (currentFunction.equals(astNode)) {
      if (complexity > threshold){
        raiseIssue();
      }
      currentFunction = null;
    }

    if (astNode.is(PythonGrammar.SUITE) && incrementsNestingLevel(astNode)) {
      nestingLevel--;
    }
  }

  private void raiseIssue() {
    String message = String.format(MESSAGE, complexity, threshold);
    PreciseIssue issue = addIssue(currentFunction.getFirstChild(PythonGrammar.FUNCNAME), message)
      .withCost(complexity - threshold);
    secondaryLocations.forEach(issue::secondary);
  }

  private boolean incrementsNestingLevel(AstNode astNode) {
    AstNode previousSibling = astNode.getPreviousSibling().getPreviousSibling();
    if (previousSibling != null && previousSibling.is(PythonKeyword.TRY, PythonKeyword.FINALLY)) {
      return false;
    }
    AstNode parent = astNode.getParent();
    if (isWrapperFunction(parent)) {
      return false;
    }
    return !parent.is(PythonGrammar.WITH_STMT, PythonGrammar.CLASSDEF)
      && (!parent.is(PythonGrammar.FUNCDEF) || !parent.equals(currentFunction));
  }

  private boolean isWrapperFunction(AstNode node) {
    if (!node.is(PythonGrammar.FUNCDEF) || node.equals(currentFunction)) {
      return false;
    }
    AstNode parentStatement = node.getParent().getParent();
    List<AstNode> ancestorStatements = node.getFirstAncestor(PythonGrammar.FUNCDEF)
      .getFirstChild(PythonGrammar.SUITE)
      .getChildren(PythonGrammar.STATEMENT);

    return ancestorStatements.stream()
      .filter(statement -> statement != parentStatement)
      .allMatch(CognitiveComplexityFunctionCheck::isSimpleReturn);
  }

  private static boolean isSimpleReturn(AstNode statement) {
    AstNode returnStatement = lookupOnlyChild(statement.getFirstChild(PythonGrammar.STMT_LIST),
      PythonGrammar.SIMPLE_STMT, PythonGrammar.RETURN_STMT);
    return returnStatement != null &&
      lookupOnlyChild(returnStatement.getFirstChild(PythonGrammar.TESTLIST),
        PythonGrammar.TEST, PythonGrammar.ATOM, PythonGrammar.NAME) != null;
  }

  private void incrementWithNesting(AstNode secondaryLocationNode) {
    int currentNodeComplexity = nestingLevel + 1;
    incrementComplexity(secondaryLocationNode, currentNodeComplexity);
  }

  private void incrementWithoutNesting(AstNode secondaryLocationNode) {
    incrementComplexity(secondaryLocationNode, 1);
  }

  private void incrementComplexity(AstNode secondaryLocationNode, int currentNodeComplexity) {
    secondaryLocations.add(IssueLocation.preciseLocation(secondaryLocationNode, secondaryMessage(currentNodeComplexity)));
    complexity += currentNodeComplexity;
  }

  private static String secondaryMessage(int complexity) {
    if (complexity == 1) {
      return "+1";

    } else{
      return String.format("+%s (incl %s for nesting)", complexity, complexity - 1);
    }
  }

  @Nullable
  private static AstNode lookupOnlyChild(@Nullable AstNode parent, AstNodeType... types) {
    if (parent == null) {
      return null;
    }
    AstNode result = parent;
    for (AstNodeType type : types) {
      List<AstNode> children = result.getChildren();
      if (children.size() != 1 || !children.get(0).is(type)) {
        return null;
      }
      result = children.get(0);
    }
    return result;
  }

}
