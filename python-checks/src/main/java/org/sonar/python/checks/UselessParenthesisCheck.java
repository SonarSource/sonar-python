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
import java.util.Collections;
import java.util.List;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.python.PythonCheck;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.api.PythonKeyword;
import org.sonar.python.api.PythonPunctuator;

@Rule(key = UselessParenthesisCheck.CHECK_KEY)
public class UselessParenthesisCheck extends PythonCheck {

  public static final String CHECK_KEY = "S1110";

  private static final String MESSAGE = "Remove those useless parentheses.";

  @Override
  public Set<AstNodeType> subscribedKinds() {
    return Collections.singleton(PythonGrammar.ATOM);
  }

  @Override
  public void visitNode(AstNode node) {
    List<AstNode> children = node.getChildren();
    boolean hasParentheses = children.size() == 3 && children.get(0).is(PythonPunctuator.LPARENTHESIS);
    if (hasParentheses) {
      AstNode child1 = children.get(1);
      if (child1.getChildren(PythonGrammar.TEST).size() == 1
        && child1.getFirstChild(PythonPunctuator.COMMA) == null
        && !child1.hasDirectChildren(PythonGrammar.COMP_FOR)) {
        AstNode test = child1.getFirstChild(PythonGrammar.TEST);
        if (!test.hasDirectChildren(PythonKeyword.IF)) {
          AstNode testChild0 = test.getFirstChild();
          if (testChild0.is(PythonGrammar.ATOM) && testChild0.getFirstChild().is(PythonPunctuator.LPARENTHESIS)) {
            addIssue(children.get(0), MESSAGE).secondary(children.get(2), null);
          }
        }
      }
    }
  }

}
