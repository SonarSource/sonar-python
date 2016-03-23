/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2016 SonarSource SA
 * mailto:contact AT sonarsource DOT com
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
import com.sonar.sslr.api.Grammar;
import org.sonar.api.server.rule.RulesDefinition;
import org.sonar.check.Priority;
import org.sonar.check.Rule;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.api.PythonPunctuator;
import org.sonar.squidbridge.annotations.ActivatedByDefault;
import org.sonar.squidbridge.annotations.SqaleConstantRemediation;
import org.sonar.squidbridge.annotations.SqaleSubCharacteristic;
import org.sonar.squidbridge.checks.SquidCheck;
import org.sonar.sslr.ast.AstSelect;

import java.util.List;

@Rule(
    key = UselessParenthesisCheck.CHECK_KEY,
    priority = Priority.MAJOR,
    name = "Useless parentheses around expressions should be removed to prevent any misunderstanding",
    tags = Tags.CONFUSING
)
@SqaleSubCharacteristic(RulesDefinition.SubCharacteristics.READABILITY)
@SqaleConstantRemediation("1min")
@ActivatedByDefault
public class UselessParenthesisCheck extends SquidCheck<Grammar> {
  public static final String CHECK_KEY = "S1110";

  @Override
  public void init() {
    subscribeTo(
        PythonGrammar.TEST,
        PythonGrammar.EXPR,
        PythonGrammar.NOT_TEST
    );
  }

  @Override
  public void visitNode(AstNode node) {
    if (node.is(PythonGrammar.NOT_TEST)) {
      visitNotTest(node);
    } else if (node.getNumberOfChildren() == 1 && node.getFirstChild().is(PythonGrammar.ATOM)) {

      AstNode atom = node.getFirstChild();
      AstSelect selectedParent = atom.select().parent().parent().parent();
      if (selectedParent.size() == 1) {
        AstNode parent = selectedParent.get(0);
        if (isKeywordException(parent)) {
          checkAtom(atom, true);
          return;
        }
      }

      checkAtom(atom, false);
    }
  }

  private boolean isKeywordException(AstNode parent) {
    if ((parent.is(PythonGrammar.RETURN_STMT) || parent.is(PythonGrammar.YIELD_EXPR)) && parent.getFirstChild(PythonGrammar.TESTLIST).getNumberOfChildren() == 1) {
      return true;
    }
    return parent.is(PythonGrammar.FOR_STMT) && parent.getFirstChild(PythonGrammar.EXPRLIST).getChildren(PythonGrammar.EXPR).size() == 1;
  }

  private void checkAtom(AstNode atom, boolean ignoreTestNumber) {
    if (violationCondition(atom, ignoreTestNumber)) {
      getContext().createLineViolation(this, "Remove those useless parentheses", atom);
    }
  }

  private boolean violationCondition(AstNode atom, boolean ignoreTestNumber) {
    List<AstNode> children = atom.getChildren();
    boolean result = children.size() == 3 && children.get(0).is(PythonPunctuator.LPARENTHESIS) && children.get(2).is(PythonPunctuator.RPARENTHESIS) && isOnASingleLine(atom);
    if (result && !ignoreTestNumber) {
      result = children.get(1).getChildren(PythonGrammar.TEST).size() == 1 && children.get(1).getFirstChild(PythonPunctuator.COMMA) == null;
    }
    return result;
  }

  private void visitNotTest(AstNode node) {
    boolean hasUselessParenthesis = node.select()
        .children(PythonGrammar.ATOM)
        .children(PythonGrammar.TESTLIST_COMP)
        .children(PythonGrammar.TEST)
        .children(PythonGrammar.ATOM, PythonGrammar.COMPARISON)
        .isNotEmpty();
    if (hasUselessParenthesis) {
      checkAtom(node.getFirstChild().getNextSibling(), false);
    }
  }

  private boolean isOnASingleLine(AstNode node) {
    return node.getTokenLine() == node.getLastToken().getLine();
  }

}
