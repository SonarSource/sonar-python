/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
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

import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonVisitorCheck;
import org.sonar.plugins.python.api.PythonVisitorContext;
import org.sonar.plugins.python.api.tree.IfStatement;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.Tree;

@Rule(key = CollapsibleIfStatementsCheck.CHECK_KEY)
public class CollapsibleIfStatementsCheck extends PythonVisitorCheck {
  public static final String CHECK_KEY = "S1066";
  private static final String MESSAGE = "Merge this if statement with the enclosing one.";

  private Set<Tree> ignored = new HashSet<>();

  @Override
  public void scanFile(PythonVisitorContext visitorContext) {
    ignored.clear();
    super.scanFile(visitorContext);
  }

  @Override
  public void visitIfStatement(IfStatement ifStatement) {
    List<Statement> statements = ifStatement.body().statements();
    if (!ifStatement.elifBranches().isEmpty()) {
      if (ifStatement.elseBranch() == null) {
        ignored.addAll(ifStatement.elifBranches().subList(0, ifStatement.elifBranches().size() - 1));
      } else {
        ignored.addAll(ifStatement.elifBranches());
      }
    }
    if (!ignored.contains(ifStatement)
      && ifStatement.elseBranch() == null
      && ifStatement.elifBranches().isEmpty()
      && statements.size() == 1
      && statements.get(0).is(Tree.Kind.IF_STMT)) {
      IfStatement singleIfChild = (IfStatement) statements.get(0);
      if (singleIfChild.isElif() || singleIfChild.elseBranch() != null || !singleIfChild.elifBranches().isEmpty()) {
        return;
      }
      addIssue(singleIfChild.keyword(), MESSAGE).secondary(ifStatement.keyword(), "enclosing");
    }
    super.visitIfStatement(ifStatement);
  }
}
