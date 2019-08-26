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
package org.sonar.python.checks;

import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import org.sonar.check.Rule;
import org.sonar.python.PythonSubscriptionCheck;
import org.sonar.python.api.tree.PyStatementListTree;
import org.sonar.python.api.tree.PyStatementTree;
import org.sonar.python.api.tree.Tree.Kind;

@Rule(key = "S1763")
public class AfterJumpStatementCheck extends PythonSubscriptionCheck {

  private static final Map<Kind, String> JUMP_KEYWORDS_BY_KIND = jumpKeywordsByKind();

  private static Map<Kind, String> jumpKeywordsByKind() {
    Map<Kind, String> map = new EnumMap<>(Kind.class);
    map.put(Kind.RETURN_STMT, "return");
    map.put(Kind.RAISE_STMT, "raise");
    map.put(Kind.BREAK_STMT, "break");
    map.put(Kind.CONTINUE_STMT, "continue");
    return map;
  }

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Kind.STATEMENT_LIST, ctx -> {
      List<PyStatementTree> statements = ((PyStatementListTree) ctx.syntaxNode()).statements();
      for (PyStatementTree statement: statements.subList(0, statements.size() - 1)) {
        String jumpKeyword = JUMP_KEYWORDS_BY_KIND.get(statement.getKind());
        if (jumpKeyword != null) {
          ctx.addIssue(statement, String.format("Remove the code after this \"%s\".", jumpKeyword));
        }
      }
    });

  }
}

