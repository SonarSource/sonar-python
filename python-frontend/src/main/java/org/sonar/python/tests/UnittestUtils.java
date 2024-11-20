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

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

public class UnittestUtils {

  private UnittestUtils() {

  }

  // Methods of unittest are the union of Python 2 and Python 3 methods:
  // https://docs.python.org/2/library/unittest.html#unittest.TestCase
  // https://docs.python.org/3/library/unittest.html#unittest.TestCase
  public static final Set<String> RUN_METHODS = Set.of("setUp", "tearDown", "setUpClass", "tearDownClass", "run", "skiptTest",
    "subTest", "debug");

  public static final Set<String> RAISE_METHODS = Set.of("assertRaises", "assertRaisesRegexp", "assertRaisesRegex");

  public static final Set<String> ASSERTIONS_METHODS = Set.of("assertEqual", "assertNotEqual", "assertTrue", "assertFalse", "assertIs",
    "assertIsNot", "assertIsNone", "assertIsNotNone", "assertIn", "assertNotIn", "assertIsInstance", "assertNotIsInstance",
    "assertAlmostEqual", "assertNotAlmostEqual", "assertGreater", "assertGreaterEqual", "assertLess", "assertLessEqual",
    "assertRegexpMatches", "assertNotRegexpMatches", "assertItemsEqual", "assertDictContainsSubset", "assertMultiLineEqual",
    "assertSequenceEqual", "assertListEqual", "assertTupleEqual", "assertSetEqual", "assertDictEqual", "assertWarns", "assertWarnsRegex",
    "assertLogs", "assertNoLogs", "assertRegex", "assertNotRegex", "assertCountEqual");


  public static final Set<String> UTIL_METHODS = Set.of("addTypeEqualityFunc", "fail", "failureException", "longMessage", "maxDiff");

  public static final Set<String> GATHER_INFO_METHODS = Set.of("countTestCases", "defaultTestResult", "id", "shortDescription", "addCleanup",
    "doCleanups", "addClassCleanup", "doClassCleanups");

  private static final Set<String> ALL_METHODS = new HashSet<>();
  private static final Set<String> ALL_ASSERT_METHODS = new HashSet<>();

  static {
    ALL_METHODS.addAll(RUN_METHODS);
    ALL_METHODS.addAll(UTIL_METHODS);
    ALL_METHODS.addAll(GATHER_INFO_METHODS);
    ALL_METHODS.addAll(ASSERTIONS_METHODS);
    ALL_METHODS.addAll(RAISE_METHODS);
    ALL_ASSERT_METHODS.addAll(ASSERTIONS_METHODS);
    ALL_ASSERT_METHODS.addAll(RAISE_METHODS);
  }

  public static Set<String> allMethods() {
    return Collections.unmodifiableSet(ALL_METHODS);
  }
  public static Set<String> allAssertMethods() {
    return Collections.unmodifiableSet(ALL_ASSERT_METHODS);
  }

  public static boolean isWithinUnittestTestCase(Tree tree) {
    List<String> parentClassesFQN = new ArrayList<>();
    ClassDef classDef = (ClassDef) TreeUtils.firstAncestorOfKind(tree, Tree.Kind.CLASSDEF);
    if (classDef != null) {
      parentClassesFQN.addAll(TreeUtils.getParentClassesFQN(classDef));
    }
    return parentClassesFQN.stream().anyMatch(parentClass -> parentClass.contains("unittest") && parentClass.contains("TestCase"));
  }
}
