/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource SA.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.python.tests;


import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.semantic.SymbolTableBuilder;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.PythonTestUtils.pythonFile;

class UnittestUtilsTest  {

  @Test
  void test_isWithinUnittestTestCase() {
    String code = "import unittest\nclass A(unittest.TestCase):  ...";
    FileInput fileInput = PythonTestUtils.parse(new SymbolTableBuilder("", pythonFile("mod1.py")), code);
    Tree tree = PythonTestUtils.getLastDescendant(fileInput, t -> t.is(Tree.Kind.ELLIPSIS));
    assertThat(UnittestUtils.isWithinUnittestTestCase(tree)).isTrue();

    code = "import unittest\nclass A(unittest.case.TestCase):  ...";
    fileInput = PythonTestUtils.parse(new SymbolTableBuilder("", pythonFile("mod1.py")), code);
    tree = PythonTestUtils.getLastDescendant(fileInput, t -> t.is(Tree.Kind.ELLIPSIS));
    assertThat(UnittestUtils.isWithinUnittestTestCase(tree)).isTrue();

    code = "import random_wrapper\nclass A(random_wrapper.unittest.TestCase):  ...";
    fileInput = PythonTestUtils.parse(new SymbolTableBuilder("", pythonFile("mod1.py")), code);
    tree = PythonTestUtils.getLastDescendant(fileInput, t -> t.is(Tree.Kind.ELLIPSIS));
    assertThat(UnittestUtils.isWithinUnittestTestCase(tree)).isTrue();

    code = "import random\nclass A(random.TestCase):  ...";
    fileInput = PythonTestUtils.parse(new SymbolTableBuilder("", pythonFile("mod1.py")), code);
    tree = PythonTestUtils.getLastDescendant(fileInput, t -> t.is(Tree.Kind.ELLIPSIS));
    assertThat(UnittestUtils.isWithinUnittestTestCase(tree)).isFalse();

    code = "import unittest\nclass A(unittest.other):  ...";
    fileInput = PythonTestUtils.parse(new SymbolTableBuilder("", pythonFile("mod1.py")), code);
    tree = PythonTestUtils.getLastDescendant(fileInput, t -> t.is(Tree.Kind.ELLIPSIS));
    assertThat(UnittestUtils.isWithinUnittestTestCase(tree)).isFalse();

    code = "...";
    fileInput = PythonTestUtils.parse(new SymbolTableBuilder("", pythonFile("mod1.py")), code);
    tree = PythonTestUtils.getLastDescendant(fileInput, t -> t.is(Tree.Kind.ELLIPSIS));
    assertThat(UnittestUtils.isWithinUnittestTestCase(tree)).isFalse();
  }

  @Test
  void all_methods() {
    assertThat(UnittestUtils.allMethods()).hasSize(59);
  }

  @Test
  void all_assert_methods() {
    assertThat(UnittestUtils.allAssertMethods()).hasSize(38);
  }
}
