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
package org.sonar.python;

import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.StatementList;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.cfg.fixpoint.ReadWriteVisitor;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.PythonTestUtils.parse;

public class ReadWriteVisitorTest {

  @Test
  public void test_usages() {
    FileInput tree = parse("x = 1; print(x); x += 3\ndef fun(): print(x)\n");
    StatementList statementList = tree.statements();
    Symbol x = ((Name) ((AssignmentStatement) statementList.statements().get(0)).lhsExpressions().get(0).expressions().get(0)).symbol();
    ReadWriteVisitor visitor = new ReadWriteVisitor();
    statementList.accept(visitor);
    assertThat(visitor.symbolToUsages().get(x).usages()).hasSize(3);
    assertThat(x.usages()).hasSize(4);
    Usage printArgument = x.usages().get(3);
    assertThat(printArgument.tree().parent().is(Tree.Kind.REGULAR_ARGUMENT)).isTrue();
    assertThat(printArgument).isNotIn(visitor.symbolToUsages().get(x).usages());
  }
}
