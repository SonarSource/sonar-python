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
package org.sonar.python.checks.utils;

import java.util.Set;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.v2.SymbolV2;
import org.sonar.plugins.python.api.symbols.v2.UsageV2;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.tree.TreeUtils;

public final class HttpxUtils {

  private static final String EVENT_HOOKS = "event_hooks";
  private static final TypeMatcher HTTPX_EXACT_ASYNC_CLIENT_MATCHER = TypeMatchers.isType("httpx.AsyncClient");
  // These documented HTTPX operations preserve the client's event hooks.
  private static final Set<String> HOOK_PRESERVING_METHODS = Set.of(
    "get", "post", "put", "patch", "delete", "head", "options", "request", "send", "stream", "build_request", "aclose");

  private HttpxUtils() {
  }

  /**
   * Recognizes a function-local direct AsyncClient binding with a later hook registration:
   * <pre>{@code
   * client = httpx.AsyncClient()
   * client.event_hooks = {"response": [hook]}
   * await client.get(url)
   * }</pre>
   * Rejects class and module bindings, aliases, and uses outside the permitted receiver operations.
   * The caller must supply the direct event_hooks assignment and its receiver.
   */
  public static boolean isStableAsyncClientReceiver(Name receiver, QualifiedExpression property, AssignmentStatement registration, SubscriptionContext ctx) {
    SymbolV2 symbol = receiver.symbolV2();
    if (symbol == null) {
      return false;
    }
    Tree binding = symbol.getSingleBindingUsage().map(UsageV2::tree).orElse(null);
    if (!(binding instanceof Name) || !(binding.parent().parent() instanceof AssignmentStatement creation) ||
      Expressions.singleAssignmentTarget(creation) != binding || !(creation.assignedValue() instanceof CallExpression constructor) ||
      !HTTPX_EXACT_ASYNC_CLIENT_MATCHER.isTrueFor(constructor.callee(), ctx) || !isRegistrationAfterCreation(creation, registration)) {
      return false;
    }
    Tree scope = TreeUtils.firstAncestorOfKind(creation, Tree.Kind.FUNCDEF, Tree.Kind.CLASSDEF, Tree.Kind.LAMBDA);
    return scope != null && scope.is(Tree.Kind.FUNCDEF) &&
      TreeUtils.firstAncestorOfKind(registration, Tree.Kind.FUNCDEF, Tree.Kind.CLASSDEF, Tree.Kind.LAMBDA) == scope && symbol.usages().stream()
      .allMatch(usage -> usage.tree() == binding || isSafeReceiverUsage(usage.tree(), property, creation, registration, scope));
  }

  /**
   * Requires registration after creation within its enclosing block, without crossing a loop:
   * <pre>{@code
   * for item in items:
   *     client = httpx.AsyncClient()
   *     client.event_hooks = {"response": [hook]}  # Accepted: a fresh client each time.
   *
   * client = httpx.AsyncClient()
   * for item in items:
   *     client.event_hooks = {"response": [hook]}  # Rejected: the client is reused.
   * }</pre>
   */
  private static boolean isRegistrationAfterCreation(AssignmentStatement creation, AssignmentStatement registration) {
    // A loop must create a fresh client before it can register hooks again.
    Tree boundary = TreeUtils.firstAncestor(registration, tree -> tree == creation.parent() || tree.is(Tree.Kind.FOR_STMT, Tree.Kind.WHILE_STMT));
    return boundary == creation.parent() && TreeUtils.getTreeByPositionComparator().compare(creation, registration) < 0;
  }

  /**
   * Permits the registration itself, direct calls to allowed methods, and hook reads before registration:
   * <pre>{@code
   * client = httpx.AsyncClient()
   * previous_hooks = client.event_hooks  # Accepted: the setter later copies the lists.
   * client.event_hooks = {"response": [hook]}
   * await client.get(url)                # Accepted: a direct call to an allowed method.
   * alias = client                      # Rejected: exposes the receiver.
   * saved_get = client.get              # Rejected: a bound method exposes the receiver.
   * later_hooks = client.event_hooks    # Rejected: exposes the registered hooks.
   * }</pre>
   * Each usage must share the creation's function or lambda scope.
   * Hook reads must follow creation within its enclosing block.
   */
  private static boolean isSafeReceiverUsage(Tree usageTree, QualifiedExpression property, AssignmentStatement creation, AssignmentStatement registration, Tree scope) {
    if (TreeUtils.firstAncestorOfKind(usageTree, Tree.Kind.FUNCDEF, Tree.Kind.CLASSDEF, Tree.Kind.LAMBDA) != scope ||
      !(usageTree.parent() instanceof QualifiedExpression qualified) ||
      qualified.qualifier() != usageTree) {
      return false;
    }
    if (qualified == property) {
      return true;
    }
    if (!EVENT_HOOKS.equals(qualified.name().name())) {
      return HOOK_PRESERVING_METHODS.contains(qualified.name().name()) && qualified.parent() instanceof CallExpression call && call.callee() == qualified;
    }
    // The setter copies the old lists. Only direct property reads before registration are safe here.
    return qualified.parent() instanceof AssignmentStatement read && read.assignedValue() == qualified &&
      TreeUtils.getTreeByPositionComparator().compare(read, registration) < 0 && TreeUtils.getTreeByPositionComparator().compare(creation, read) < 0 &&
      TreeUtils.firstAncestor(read, tree -> tree == creation.parent()) != null;
  }

}
