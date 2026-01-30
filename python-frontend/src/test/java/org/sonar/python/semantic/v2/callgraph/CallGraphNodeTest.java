/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
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
package org.sonar.python.semantic.v2.callgraph;

import java.lang.ref.WeakReference;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.tree.Tree;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;

class CallGraphNodeTest {
  @Test
  void testEqualsAndHashCode() {
    CallGraphNode fun1 = new CallGraphNode("function1", new WeakReference<>(null));
    CallGraphNode fun1Copy = new CallGraphNode("function1", new WeakReference<>(null));
    CallGraphNode fun2 = new CallGraphNode("function2", new WeakReference<>(null));

    Tree fun1Tree = mock();
    Tree fun2Tree = mock();
    CallGraphNode fun1WithTree = new CallGraphNode("function1", new WeakReference<>(fun1Tree));
    CallGraphNode fun1WithTreeCopy = new CallGraphNode("function1", new WeakReference<>(fun1Tree));
    CallGraphNode fun2WithTree = new CallGraphNode("function2", new WeakReference<>(fun2Tree));

    Object otherObject = new Object();

    assertThat(fun1Copy)
      .isEqualTo(fun1)
      .hasSameHashCodeAs(fun1);
    
    assertThat(fun1)
      .isEqualTo(fun1);

    assertThat(fun2)
      .isNotEqualTo(fun1)
      .doesNotHaveSameHashCodeAs(fun1);

    assertThat(fun1)
      .isNotEqualTo(otherObject)
      .doesNotHaveSameHashCodeAs(otherObject);

    assertThat(fun1WithTree)
      .isEqualTo(fun1WithTreeCopy)
      .hasSameHashCodeAs(fun1WithTreeCopy);

    assertThat(fun2WithTree)
      .isNotEqualTo(fun1WithTree)
      .doesNotHaveSameHashCodeAs(fun1WithTree);
  }

  @Test
  void testWeakReferenceEquality() {
    Tree fun1Tree = mock();
    var fun1TreeRef = new WeakReference<>(fun1Tree);
    CallGraphNode fun1WithTree = new CallGraphNode("function1", fun1TreeRef);

    Tree fun3Tree = mock();
    var fun3TreeRef = new WeakReference<>(fun3Tree);
    CallGraphNode function1WithTree3 = new CallGraphNode("function1", fun3TreeRef);
    assertThat(function1WithTree3)
      .isNotEqualTo(fun1WithTree);

    // simulate garbage collection
    fun3TreeRef.clear();
    fun1TreeRef.clear();

    assertThat(fun1WithTree.tree()).isEmpty();
    assertThat(function1WithTree3.tree()).isEmpty();

    assertThat(function1WithTree3)
      .isEqualTo(fun1WithTree);
  }

}
