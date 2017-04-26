/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2017 SonarSource SA
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

import com.google.common.collect.ImmutableSet;
import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.AstNodeType;
import java.util.List;
import java.util.Set;
import java.util.List;

import org.sonar.check.Priority;
import org.sonar.check.Rule;
import org.sonar.python.PythonCheck;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.api.PythonKeyword;
import org.sonar.python.api.PythonPunctuator;
import org.sonar.squidbridge.annotations.ActivatedByDefault;
import org.sonar.squidbridge.annotations.SqaleConstantRemediation;

import com.sonar.sslr.api.AstNode;

@Rule(
    key = UselessParenthesisCheck.CHECK_KEY,
    priority = Priority.MAJOR,
    name = "Redundant parentheses around expressions should be removed",
    tags = Tags.CONFUSING
)
@SqaleConstantRemediation("1min")
@ActivatedByDefault
public class UselessParenthesisCheck extends PythonCheck {

  public static final String CHECK_KEY = "S1110";

  private static final String MESSAGE = "Remove those useless parentheses.";

  @Override
  public Set<AstNodeType> subscribedKinds() {
    return ImmutableSet.of(
        PythonGrammar.TEST,
        PythonGrammar.EXPR,
        PythonGrammar.A_EXPR,
        PythonGrammar.M_EXPR,
        PythonGrammar.NOT_TEST
    );
  }

  @Override
  public void visitNode(AstNode node) {
    if (node.is(PythonGrammar.NOT_TEST)) {
      checkAtom(node.getFirstChild().getNextSibling());
    } else {
      node.getChildren(PythonGrammar.ATOM).forEach(this::checkAtom);
    }
  }

  private void checkAtom(AstNode atom) {
    List<AstNode> children = atom.getChildren();
    boolean hasParentheses = children.size() == 3 && children.get(0).is(PythonPunctuator.LPARENTHESIS);
    if (hasParentheses) {
      AstNode child1 = children.get(1);
      if(child1.getChildren(PythonGrammar.TEST).size() == 1 && child1.getFirstChild(PythonPunctuator.COMMA) == null) {
        AstNode test = child1.getChildren(PythonGrammar.TEST).get(0);
        if (test.getChildren(PythonKeyword.IF).isEmpty()) {
          AstNode testChild0 = test.getChildren().get(0);
          if (testChild0.is(PythonGrammar.ATOM) && testChild0.getFirstChild().is(PythonPunctuator.LPARENTHESIS)) {
            addIssue(children.get(0), MESSAGE).secondary(children.get(2), null);
          }
        }
      }
    }
  }

}
