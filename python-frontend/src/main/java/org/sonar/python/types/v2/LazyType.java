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
package org.sonar.python.types.v2;

import java.util.ArrayDeque;
import java.util.Optional;
import java.util.Queue;
import java.util.function.Consumer;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.python.semantic.v2.SymbolsModuleTypeProvider;

public class LazyType implements PythonType {

  String fullyQualifiedName;
  private final Queue<Consumer<PythonType>> consumers;
  private final SymbolsModuleTypeProvider symbolsModuleTypeProvider;

  public LazyType(String fullyQualifiedName, SymbolsModuleTypeProvider symbolsModuleTypeProvider) {
    this.fullyQualifiedName = fullyQualifiedName;
    this.symbolsModuleTypeProvider = symbolsModuleTypeProvider;
    consumers = new ArrayDeque<>();
  }

  public String fullyQualifiedName() {
    return fullyQualifiedName;
  }

  public LazyType addConsumer(Consumer<PythonType> consumer) {
    consumers.add(consumer);
    return this;
  }

  public LazyType resolve(PythonType type) {
    consumers.forEach(c -> c.accept(type));
    consumers.clear();
    return this;
  }

  public PythonType resolve() {
    PythonType resolvedType = symbolsModuleTypeProvider.resolveLazyType(this);
    consumers.forEach(c -> c.accept(resolvedType));
    consumers.clear();
    return resolvedType;
  }

  @Override
  public PythonType unwrappedType() {
    return this.resolve().unwrappedType();
  }

  @Override
  public String name() {
    return this.resolve().name();
  }

  @Override
  public Optional<String> displayName() {
    return this.resolve().displayName();
  }

  @Override
  public Optional<String> instanceDisplayName() {
    return this.resolve().instanceDisplayName();
  }

  @Override
  public boolean isCompatibleWith(PythonType another) {
    return this.resolve().isCompatibleWith(another);
  }

  @Override
  public String key() {
    return this.resolve().key();
  }

  @Override
  public Optional<PythonType> resolveMember(String memberName) {
    return this.resolve().resolveMember(memberName);
  }

  @Override
  public TriBool hasMember(String memberName) {
    return this.resolve().hasMember(memberName);
  }

  @Override
  public Optional<LocationInFile> definitionLocation() {
    return this.resolve().definitionLocation();
  }

  @Override
  public TypeSource typeSource() {
    return this.resolve().typeSource();
  }
}
