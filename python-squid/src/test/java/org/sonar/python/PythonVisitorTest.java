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
package org.sonar.python;

import com.google.common.collect.ImmutableSet;
import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.AstNodeType;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import org.junit.Test;
import org.sonar.python.api.PythonGrammar;

import static org.assertj.core.api.Assertions.assertThat;

public class PythonVisitorTest {

  @Test
  public void test() throws Exception {
    TestVisitor visitor = new TestVisitor();
    TestPythonVisitorRunner.scanFile(new File("src/test/resources/visitor.py"), visitor);
    assertThat(visitor.atomValues).containsExactly("foo", "\"x\"", "42");
    assertThat(visitor.fileName).isEqualTo("visitor.py");
  }

  public class TestVisitor extends PythonVisitor {

    private List<String> atomValues = new ArrayList<>();
    private String fileName;

    @Override
    public Set<AstNodeType> subscribedKinds() {
      return ImmutableSet.of(PythonGrammar.ATOM);
    }

    @Override
    public void visitNode(AstNode node) {
      atomValues.add(node.getTokenValue());
      fileName = getContext().pythonFile().fileName();
    }

  }

}
