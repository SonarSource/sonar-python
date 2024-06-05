/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
package org.sonar.python.semantic.v2.typeshed;

import java.util.ArrayDeque;
import java.util.List;
import java.util.Queue;
import java.util.function.Consumer;
import org.sonar.python.types.v2.PythonType;

public class PromiseType implements PythonType {
  private final Queue<Consumer<PythonType>> consumers;
  private final String fqn;

  public PromiseType(String fqn) {
    this.fqn = fqn;
    consumers = new ArrayDeque<>();
  }

  public PromiseType addConsumer(Consumer<PythonType> consumer) {
    consumers.add(consumer);
    return this;
  }

  public PromiseType resolve(PythonType type) {
    consumers.forEach(c -> c.accept(type));
    consumers.clear();
    return this;
  }

  public static Consumer<PythonType> collectionTypeResolver(PromiseType promiseType, List<PythonType> collection) {
    return resolved -> collection.replaceAll(s -> s == promiseType ? resolved : s);
  }
}
