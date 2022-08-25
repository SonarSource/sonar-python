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

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

public class UnittestUtils {

  private UnittestUtils() {

  }

  public static final Set<String> ASSERTIONS_METHODS = Set.of("assertEqual",
    "assertNotEqual", "assertTrue", "assertFalse", "assertIs", "assertIsNot", "assertIsNone", "assertIsNotNone", "assertIn",
    "assertNotIn", "assertIsInstance", "assertNotIsInstance", "assertRaises", "assertRaisesRegexp", "assertAlmostEqual",
    "assertNotAlmostEqual", "assertGreater", "assertGreaterEqual", "assertLess", "assertLessEqual", "assertRegexpMatches",
    "assertNotRegexpMatches", "assertItemsEqual", "assertDictContainsSubset", "assertMultiLineEqual", "assertSequenceEqual",
    "assertListEqual", "assertTupleEqual", "assertSetEqual", "assertDictEqual");

  public static boolean isWithinUnittestTestCase(Tree tree) {
    List<String> parentClassesFQN = new ArrayList<>();
    ClassDef classDef = (ClassDef) TreeUtils.firstAncestorOfKind(tree, Tree.Kind.CLASSDEF);
    if (classDef != null) {
      parentClassesFQN.addAll(TreeUtils.getParentClassesFQN(classDef));
    }
    return parentClassesFQN.stream().anyMatch(parentClass -> parentClass.contains("unittest") && parentClass.contains("TestCase"));
  }
}
