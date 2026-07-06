/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.python.checks.tests;

import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.net.URI;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RaiseStatement;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.utils.PythonCheckVerifier;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GroupSimilarTestsParameterizedCheckTest {

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/tests/test_groupSimilarTestsParameterized.py", new GroupSimilarTestsParameterizedCheck());
  }

  @Test
  void testNonTestFileName() {
    PythonCheckVerifier.verifyNoIssue("src/test/resources/checks/tests/groupSimilarTestsParameterized.py", new GroupSimilarTestsParameterizedCheck());
  }

  @Test
  void testUnderscoreTestSuffixFileName() {
    PythonCheckVerifier.verify("src/test/resources/checks/tests/groupSimilarTestsParameterized_test.py", new GroupSimilarTestsParameterizedCheck());
  }

  @Test
  void testEmptyFile() {
    PythonCheckVerifier.verifyNoIssue("src/test/resources/checks/tests/empty_groupSimilarTestsParameterized.py", new GroupSimilarTestsParameterizedCheck());
  }

  @Test
  void scope() {
    assertEquals(PythonCheck.CheckScope.ALL, new GroupSimilarTestsParameterizedCheck().scope());
  }

  @Test
  void same_test_kind_can_differ_between_pytest_and_unittest() throws Exception {
    FunctionDef unittestFunction = functionDef(unittestClassDef());
    FunctionDef pytestFunction = functionDef(null);

    assertFalse((boolean) invokePrivateStaticMethod("sameTestKind", new Class<?>[] {FunctionDef.class, FunctionDef.class}, unittestFunction, pytestFunction));
  }

  @Test
  void similar_tests_reject_mixed_test_kinds() throws Exception {
    FunctionDef unittestFunction = functionDef("test_variant_1", unittestClassDef());
    FunctionDef pytestFunction = functionDef("test_variant_1", null);

    assertFalse((boolean) invokePrivateStaticMethod("areSimilarTests", new Class<?>[] {FunctionDef.class, FunctionDef.class}, unittestFunction, pytestFunction));
  }

  @Test
  void not_implemented_error_raise_rejects_raise_without_expression() throws Exception {
    RaiseStatement bareRaise = proxy(RaiseStatement.class, Map.of("expressions", List.of()));

    assertFalse((boolean) invokePrivateStaticMethod("isNotImplementedErrorRaise",
      new Class<?>[] {org.sonar.plugins.python.api.tree.Statement.class, SubscriptionContext.class},
      bareRaise, dummySubscriptionContext()));
  }

  @Test
  void difference_counter_handles_defensive_null_paths() throws Exception {
    Object counter = newDifferenceCounter();
    Method areSimilar = differenceCounterMethod("areSimilar", Tree.class, Tree.class);

    assertTrue((boolean) areSimilar.invoke(counter, null, null));

    Tree tree = leaf(Tree.Kind.NAME, "value", null);
    assertFalse((boolean) areSimilar.invoke(counter, tree, null));
  }

  @Test
  void difference_counter_handles_leafs_without_tokens_or_parents() throws Exception {
    Method areSimilar = differenceCounterMethod("areSimilar", Tree.class, Tree.class);
    Method compareLeaves = differenceCounterMethod("compareLeaves", Tree.class, Tree.class);

    Tree leafWithoutToken1 = leaf(Tree.Kind.PASS_STMT, null, null);
    Tree leafWithoutToken2 = leaf(Tree.Kind.PASS_STMT, null, null);
    assertTrue((boolean) areSimilar.invoke(newDifferenceCounter(), leafWithoutToken1, leafWithoutToken2));

    Tree nonParameterizableLeaf1 = leaf(Tree.Kind.PASS_STMT, "left", null);
    Tree nonParameterizableLeaf2 = leaf(Tree.Kind.PASS_STMT, "right", null);
    assertFalse((boolean) areSimilar.invoke(newDifferenceCounter(), nonParameterizableLeaf1, nonParameterizableLeaf2));

    Tree parameterizableLeaf1 = leaf(Tree.Kind.NUMERIC_LITERAL, "1", null);
    Tree parameterizableLeaf2 = leaf(Tree.Kind.NUMERIC_LITERAL, "2", null);
    assertTrue((boolean) areSimilar.invoke(newDifferenceCounter(), parameterizableLeaf1, parameterizableLeaf2));

    Tree leftNullTokenLeaf = leaf(Tree.Kind.PASS_STMT, null, null);
    Tree rightTokenLeaf = leaf(Tree.Kind.PASS_STMT, "right", null);
    assertFalse((boolean) compareLeaves.invoke(newDifferenceCounter(), leftNullTokenLeaf, rightTokenLeaf));

    Tree rightNonParameterizableLeaf = leaf(Tree.Kind.PASS_STMT, "right", null);
    assertFalse((boolean) compareLeaves.invoke(newDifferenceCounter(), parameterizableLeaf1, rightNonParameterizableLeaf));
  }

  @Test
  void difference_counter_handles_left_null_and_nested_trees() throws Exception {
    Method areSimilar = differenceCounterMethod("areSimilar", Tree.class, Tree.class);

    Tree tree = leaf(Tree.Kind.PASS_STMT, "value", null);
    assertFalse((boolean) areSimilar.invoke(newDifferenceCounter(), null, tree));

    Tree nestedLeft = tree(Tree.Kind.TUPLE, List.of(leaf(Tree.Kind.NUMERIC_LITERAL, "1", null)));
    Tree nestedRight = tree(Tree.Kind.TUPLE, List.of(leaf(Tree.Kind.NUMERIC_LITERAL, "1", null)));
    assertTrue((boolean) areSimilar.invoke(newDifferenceCounter(), nestedLeft, nestedRight));
  }

  private static Object newDifferenceCounter() throws Exception {
    Constructor<?> constructor = Class.forName("org.sonar.python.checks.tests.GroupSimilarTestsParameterizedCheck$DifferenceCounter").getDeclaredConstructor();
    constructor.setAccessible(true);
    return constructor.newInstance();
  }

  private static Method differenceCounterMethod(String name, Class<?>... parameterTypes) throws Exception {
    Method method = Class.forName("org.sonar.python.checks.tests.GroupSimilarTestsParameterizedCheck$DifferenceCounter")
      .getDeclaredMethod(name, parameterTypes);
    method.setAccessible(true);
    return method;
  }

  private static Object invokePrivateStaticMethod(String name, Class<?>[] parameterTypes, Object... arguments) throws Exception {
    Method method = GroupSimilarTestsParameterizedCheck.class.getDeclaredMethod(name, parameterTypes);
    method.setAccessible(true);
    return method.invoke(null, arguments);
  }

  private static ClassDef unittestClassDef() {
    Symbol parentClass = proxy(Symbol.class, Map.of(
      "fullyQualifiedName", "unittest.case.TestCase"));

    ClassSymbol classSymbol = proxy(ClassSymbol.class, Map.of(
      "kind", Symbol.Kind.CLASS,
      "superClasses", List.of(parentClass)));

    Map<String, Object> classNameConfig = new HashMap<>();
    classNameConfig.put("symbol", classSymbol);
    classNameConfig.put("kind", Tree.Kind.NAME);
    classNameConfig.put("children", List.of());
    classNameConfig.put("parent", null);
    classNameConfig.put("firstToken", null);
    Name className = proxy(Name.class, classNameConfig);

    Map<String, Object> classDefConfig = new HashMap<>();
    classDefConfig.put("kind", Tree.Kind.CLASSDEF);
    classDefConfig.put("name", className);
    classDefConfig.put("parent", null);
    classDefConfig.put("children", List.of());
    classDefConfig.put("firstToken", null);
    return proxy(ClassDef.class, classDefConfig);
  }

  private static FunctionDef functionDef(Tree parent) {
    return functionDef("test_variant", parent);
  }

  private static FunctionDef functionDef(String name, Tree parent) {
    Map<String, Object> nameConfig = new HashMap<>();
    nameConfig.put("kind", Tree.Kind.NAME);
    nameConfig.put("name", name);
    nameConfig.put("children", List.of());
    nameConfig.put("parent", null);
    nameConfig.put("firstToken", null);
    Name functionName = proxy(Name.class, nameConfig);

    Map<String, Object> functionConfig = new HashMap<>();
    functionConfig.put("kind", Tree.Kind.FUNCDEF);
    functionConfig.put("name", functionName);
    functionConfig.put("parent", parent);
    functionConfig.put("children", List.of());
    functionConfig.put("firstToken", null);
    functionConfig.put("parameters", null);
    functionConfig.put("body", null);
    return proxy(FunctionDef.class, functionConfig);
  }

  private static Tree leaf(Tree.Kind kind, String tokenValue, Tree parent) {
    Token token = tokenValue == null ? null : proxy(Token.class, Map.of("value", tokenValue));
    Map<String, Object> treeConfig = new HashMap<>();
    treeConfig.put("kind", kind);
    treeConfig.put("children", List.of());
    treeConfig.put("parent", parent);
    treeConfig.put("firstToken", token);
    return proxy(Tree.class, treeConfig);
  }

  private static Tree tree(Tree.Kind kind, List<Tree> children) {
    Map<String, Object> treeConfig = new HashMap<>();
    treeConfig.put("kind", kind);
    treeConfig.put("children", children);
    treeConfig.put("parent", null);
    treeConfig.put("firstToken", null);
    return proxy(Tree.class, treeConfig);
  }

  private static SubscriptionContext dummySubscriptionContext() {
    return proxy(SubscriptionContext.class, Map.of(
      "pythonFile", new org.sonar.plugins.python.api.PythonFile() {
        @Override
        public String content() {
          return "";
        }

        @Override
        public String fileName() {
          return "test_groupSimilarTestsParameterized.py";
        }

        @Override
        public URI uri() {
          return URI.create("file:///test_groupSimilarTestsParameterized.py");
        }

        @Override
        public String key() {
          return "test_groupSimilarTestsParameterized.py";
        }
      }));
  }

  @SuppressWarnings("unchecked")
  private static <T> T proxy(Class<T> type, Map<String, Object> values) {
    Map<String, Object> config = new HashMap<>(values);
    return (T) Proxy.newProxyInstance(type.getClassLoader(), new Class<?>[] {type}, (proxy, method, args) -> switch (method.getName()) {
      case "getKind" -> config.get("kind");
      case "children" -> config.getOrDefault("children", List.of());
      case "parent" -> config.get("parent");
      case "firstToken" -> config.get("firstToken");
      case "name" -> config.get("name");
      case "parameters" -> config.get("parameters");
      case "body" -> config.get("body");
      case "symbol" -> config.get("symbol");
      case "kind" -> config.get("kind");
      case "superClasses" -> config.getOrDefault("superClasses", List.of());
      case "fullyQualifiedName" -> config.get("fullyQualifiedName");
      case "expressions" -> config.getOrDefault("expressions", List.of());
      case "pythonFile" -> config.get("pythonFile");
      case "value" -> config.get("value");
      case "is" -> Arrays.stream((Tree.Kind[]) args[0]).anyMatch(config.get("kind")::equals);
      case "hashCode" -> System.identityHashCode(proxy);
      case "equals" -> proxy == args[0];
      case "toString" -> type.getSimpleName() + config;
      default -> defaultValue(method.getReturnType());
    });
  }

  private static Object defaultValue(Class<?> returnType) {
    if (!returnType.isPrimitive()) {
      return null;
    }
    if (boolean.class.equals(returnType)) {
      return false;
    }
    if (char.class.equals(returnType)) {
      return '\0';
    }
    return 0;
  }
}
