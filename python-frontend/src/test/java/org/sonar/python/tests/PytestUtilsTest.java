/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2022 SonarSource SA
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
package org.sonar.python.tests;

import org.junit.Test;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.semantic.SymbolTableBuilder;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.PythonTestUtils.pythonFile;

public class PytestUtilsTest {

  @Test
  public void test_isPytest() {
    String code = "import pytest\ndef test():\n\tpytest.raises(ZeroDivisionError)";
    FileInput fileInput = PythonTestUtils.parse(new SymbolTableBuilder("", pythonFile("mod1.py")), code);
    Tree tree = PythonTestUtils.getLastDescendant(fileInput, t -> t.is(Tree.Kind.QUALIFIED_EXPR));
    assertThat(PytestUtils.isPytest((QualifiedExpression) tree)).isTrue();

    code = "from random_wrapper import pytest\ndef test():\n\tpytest.raises(ZeroDivisionError)";
    fileInput = PythonTestUtils.parse(new SymbolTableBuilder("", pythonFile("mod1.py")), code);
    tree = PythonTestUtils.getLastDescendant(fileInput, t -> t.is(Tree.Kind.QUALIFIED_EXPR));
    assertThat(PytestUtils.isPytest((QualifiedExpression) tree)).isTrue();

    code = "import random\ndef test():\n\trandom.pytest.raises(ZeroDivisionError)";
    fileInput = PythonTestUtils.parse(new SymbolTableBuilder("", pythonFile("mod1.py")), code);
    tree = PythonTestUtils.getLastDescendant(fileInput, t -> t.is(Tree.Kind.QUALIFIED_EXPR));
    assertThat(PytestUtils.isPytest((QualifiedExpression) tree)).isFalse();
  }
}
