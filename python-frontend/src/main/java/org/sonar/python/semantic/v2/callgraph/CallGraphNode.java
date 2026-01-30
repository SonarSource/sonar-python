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
import java.util.Objects;
import java.util.Optional;
import org.sonar.plugins.python.api.tree.Tree;

public class CallGraphNode {
  private final String fqn;
  private final WeakReference<Tree> tree;

  public CallGraphNode(String fqn, WeakReference<Tree> tree) {
    this.fqn = fqn;
    this.tree = tree;
  }

  public CallGraphNode(String fqn) {
    this(fqn, new WeakReference<>(null));
  }

  public String fqn() {
    return fqn;
  }

  public Optional<Tree> tree() {
    return Optional.ofNullable(tree.get());
  }

  @Override
  public int hashCode() {
    return Objects.hash(fqn, tree());
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof CallGraphNode other)) {
      return false;
    }
    return Objects.equals(fqn, other.fqn) && Objects.equals(tree(), other.tree());
  }

  @Override
  public String toString() {
    return "CallGraphNode{" +
      "fqn='" + fqn + '\'' +
      ", tree=" + tree.get() +
      '}';
  }

}

