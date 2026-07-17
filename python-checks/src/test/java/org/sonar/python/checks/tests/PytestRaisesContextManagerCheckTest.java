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

import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.utils.PythonCheckVerifier;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PytestRaisesContextManagerCheckTest {

  @Test
  void scope() {
    assertEquals(PythonCheck.CheckScope.ALL, new PytestRaisesContextManagerCheck().scope());
  }

  @Test
  void sample() {
    PythonCheckVerifier.verify("src/test/resources/checks/tests/pytestRaisesContextManager.py", new PytestRaisesContextManagerCheck());
  }

  /**
   * Defensive coverage: {@code FunctionDef.parameters()} is null for some incomplete ASTs
   * (e.g. lambda-like / synthetic trees). Real Python samples always produce a ParameterList,
   * so this branch cannot be exercised via {@link #sample()}.
   */
  @Test
  void parameterSymbol_nullParameters() throws Exception {
    Map<String, Object> config = new HashMap<>();
    config.put("kind", Tree.Kind.FUNCDEF);
    config.put("parameters", null);
    config.put("children", List.of());
    config.put("parent", null);
    config.put("firstToken", null);
    FunctionDef functionDef = proxy(FunctionDef.class, config);
    Optional<?> result = (Optional<?>) invokePrivateStatic("parameterSymbol",
      new Class<?>[] {FunctionDef.class, String.class}, functionDef, "expectation");
    assertTrue(result.isEmpty());
  }

  /**
   * Defensive coverage: early return when the call has no decorator ancestor
   * ({@code enclosingParametrizeCall} is null). That path is not reachable through
   * {@link #sample()} because the method is only invoked for calls already found under
   * a parametrize decorator in normal analysis, but the null-check is still required.
   */
  @Test
  void isParametrizeInjectedAndUsedInWith_noDecoratorAncestor() throws Exception {
    Map<String, Object> config = new HashMap<>();
    config.put("kind", Tree.Kind.CALL_EXPR);
    config.put("children", List.of());
    config.put("parent", null);
    config.put("firstToken", null);
    config.put("arguments", List.of());
    CallExpression raisesCall = proxy(CallExpression.class, config);
    boolean result = (boolean) invokePrivateStatic("isParametrizeInjectedAndUsedInWith",
      new Class<?>[] {CallExpression.class, org.sonar.plugins.python.api.SubscriptionContext.class},
      raisesCall, null);
    assertFalse(result);
  }

  private static Object invokePrivateStatic(String methodName, Class<?>[] parameterTypes, Object... arguments) throws Exception {
    Method method = PytestRaisesContextManagerCheck.class.getDeclaredMethod(methodName, parameterTypes);
    method.setAccessible(true);
    return method.invoke(null, arguments);
  }

  private static <T> T proxy(Class<T> type, Map<String, Object> values) {
    return type.cast(Proxy.newProxyInstance(type.getClassLoader(), new Class<?>[] {type}, (proxy, method, args) -> {
      String name = method.getName();
      if (values.containsKey(name)) {
        return values.get(name);
      }
      if ("hashCode".equals(name)) {
        return System.identityHashCode(proxy);
      }
      if ("equals".equals(name)) {
        return proxy == args[0];
      }
      if ("toString".equals(name)) {
        return "proxy(" + type.getSimpleName() + ")";
      }
      Class<?> returnType = method.getReturnType();
      if (returnType.equals(boolean.class)) {
        return false;
      }
      if (returnType.equals(int.class)) {
        return 0;
      }
      if (List.class.isAssignableFrom(returnType)) {
        return List.of();
      }
      return null;
    }));
  }
}
