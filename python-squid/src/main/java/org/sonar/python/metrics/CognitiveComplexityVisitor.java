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

import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.AstNodeType;
import java.util.Arrays;
import java.util.Collections;
import java.util.Deque;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import javax.annotation.Nullable;
import org.sonar.python.PythonVisitor;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.api.PythonKeyword;
import org.sonar.python.api.PythonPunctuator;

public class CognitiveComplexityVisitor extends PythonVisitor {

  private int complexity = 0;
  private Deque<NestingLevel> nestingLevelStack = new LinkedList<>();

  @Nullable
  private final SecondaryLocationConsumer secondaryLocationConsumer;

  public interface SecondaryLocationConsumer {
    void consume(AstNode node, String message);
  }

  CognitiveComplexityVisitor(@Nullable SecondaryLocationConsumer secondaryLocationConsumer) {
    this.secondaryLocationConsumer = secondaryLocationConsumer;
    nestingLevelStack.push(new NestingLevel());
  }

  public static int complexity(AstNode node, @Nullable SecondaryLocationConsumer secondaryLocationConsumer) {
    CognitiveComplexityVisitor visitor = new CognitiveComplexityVisitor(secondaryLocationConsumer);
    visitor.scanNode(node);
    return visitor.complexity;
  }

  public int getComplexity() {
    return complexity;
  }

  @Override
  public Set<AstNodeType> subscribedKinds() {
    return Collections.unmodifiableSet(new HashSet<>(Arrays.asList(
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
      PythonGrammar.CLASSDEF,
      PythonGrammar.SUITE)));
  }

  @Override
  public void visitNode(AstNode astNode) {
    if (astNode.is(PythonGrammar.FUNCDEF, PythonGrammar.CLASSDEF)) {
      nestingLevelStack.push(new NestingLevel(nestingLevelStack.peek(), astNode));
    } else if (astNode.is(PythonGrammar.SUITE)) {
      if (isSuiteIncrementsNestingLevel(astNode)) {
        nestingLevelStack.peek().increment();
      }
    } else if (astNode.is(PythonGrammar.IF_STMT, PythonGrammar.WHILE_STMT, PythonGrammar.FOR_STMT, PythonGrammar.EXCEPT_CLAUSE)) {
      incrementWithNesting(astNode.getFirstChild());
    } else if (astNode.is(PythonKeyword.ELIF) || (astNode.is(PythonKeyword.ELSE) && astNode.getNextSibling().is(PythonPunctuator.COLON))) {
      incrementWithoutNesting(astNode);
    } else if (astNode.is(PythonGrammar.AND_TEST, PythonGrammar.OR_TEST)) {
      incrementWithoutNesting(astNode.getFirstChild(PythonKeyword.AND, PythonKeyword.OR));
    } else if (astNode.is(PythonGrammar.TEST) && astNode.hasDirectChildren(PythonKeyword.IF)) {
      // conditional expression
      incrementWithNesting(astNode.getFirstChild(PythonKeyword.IF));
      nestingLevelStack.peek().increment();
    }
  }

  @Override
  public void leaveNode(AstNode astNode) {
    if (astNode.is(PythonGrammar.FUNCDEF, PythonGrammar.CLASSDEF)) {
      nestingLevelStack.pop();
    } else if (astNode.is(PythonGrammar.SUITE)) {
      if (isSuiteIncrementsNestingLevel(astNode)) {
        nestingLevelStack.peek().decrement();
      }
    } else if (astNode.is(PythonGrammar.TEST) && astNode.hasDirectChildren(PythonKeyword.IF)) {
      // conditional expression
      nestingLevelStack.peek().decrement();
    }
  }

  private static boolean isSuiteIncrementsNestingLevel(AstNode astNode) {
    AstNode previousSibling = astNode.getPreviousSibling().getPreviousSibling();
    if (previousSibling.is(PythonKeyword.TRY, PythonKeyword.FINALLY)) {
      return false;
    }
    return !astNode.getParent().is(PythonGrammar.CLASSDEF, PythonGrammar.FUNCDEF, PythonGrammar.WITH_STMT);
  }

  private void incrementWithNesting(AstNode secondaryLocationNode) {
    incrementComplexity(secondaryLocationNode, 1 + nestingLevelStack.peek().level());
  }

  private void incrementWithoutNesting(AstNode secondaryLocationNode) {
    incrementComplexity(secondaryLocationNode, 1);
  }

  private void incrementComplexity(AstNode secondaryLocationNode, int currentNodeComplexity) {
    if (secondaryLocationConsumer != null) {
      secondaryLocationConsumer.consume(secondaryLocationNode, secondaryMessage(currentNodeComplexity));
    }
    complexity += currentNodeComplexity;
  }

  private static String secondaryMessage(int complexity) {
    if (complexity == 1) {
      return "+1";
    } else {
      return String.format("+%s (incl %s for nesting)", complexity, complexity - 1);
    }
  }

  private static class NestingLevel {

    @Nullable
    private AstNode astNode;
    private int level;

    private NestingLevel() {
      astNode = null;
      level = 0;
    }

    private NestingLevel(NestingLevel parent, AstNode astNode) {
      this.astNode = astNode;
      if (astNode.is(PythonGrammar.FUNCDEF)) {
        if (parent.isWrapperFunction(astNode)) {
          level = parent.level;
        } else if (parent.isFunction()) {
          level = parent.level + 1;
        } else {
          level = 0;
        }
      } else {
        // PythonGrammar.CLASSDEF
        level = 0;
      }
    }

    private boolean isFunction() {
      return astNode != null && astNode.is(PythonGrammar.FUNCDEF);
    }

    private boolean isWrapperFunction(AstNode childFunction) {
      if(astNode != null && astNode.is(PythonGrammar.FUNCDEF)) {
        AstNode childStatement = childFunction.getParent().getParent();
        return astNode.getFirstChild(PythonGrammar.SUITE)
          .getChildren(PythonGrammar.STATEMENT)
          .stream()
          .filter(statement -> statement != childStatement)
          .allMatch(NestingLevel::isSimpleReturn);
      }
      return false;
    }

    private static boolean isSimpleReturn(AstNode statement) {
      AstNode returnStatement = lookupOnlyChild(statement.getFirstChild(PythonGrammar.STMT_LIST),
        PythonGrammar.SIMPLE_STMT, PythonGrammar.RETURN_STMT);
      return returnStatement != null &&
        lookupOnlyChild(returnStatement.getFirstChild(PythonGrammar.TESTLIST),
          PythonGrammar.TEST, PythonGrammar.ATOM, PythonGrammar.NAME) != null;
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

    private int level() {
      return level;
    }

    private void increment() {
      level++;
    }

    private void decrement() {
      level--;
    }

  }

}
