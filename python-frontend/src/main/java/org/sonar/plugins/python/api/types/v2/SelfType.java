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
package org.sonar.plugins.python.api.types.v2;

import java.util.Objects;
import java.util.Optional;
import javax.annotation.Nullable;
import org.sonar.api.Beta;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.plugins.python.api.TriBool;

/**
 * Represents typing.Self or typing_extensions.Self type as defined in PEP 673.
 * <p>
 * The Self type is used to indicate that a method returns an instance of the same type as the enclosing class.
 * This is particularly useful in methods that return instances of the class, such as constructors or builder methods.
 * <p>
 * Important invariants:
 * <ul>
 * <li>A SelfType must NOT wrap an ObjectType - instead the ObjectType should wrap the SelfType</li>
 * <li>A SelfType must NOT wrap a UnionType - instead the UnionType should wrap the SelfType</li>
 * </ul>
 * <p>
 * Use the static factory method {@link #of(PythonType)} to create instances, which ensures these invariants are maintained.
 * 
 * @see <a href="https://peps.python.org/pep-0673/">PEP 673 – Self Type</a>
 * @see <a href="https://typing.python.org/en/latest/spec/generics.html#self">Typing Spec - Self</a>
 * @see <a href="https://docs.python.org/3/library/typing.html#typing.Self">Python Docs - typing.Self</a>
 */
@Beta
public final class SelfType implements PythonType {
  private final PythonType innerType;

  private SelfType(PythonType innerType) {
    this.innerType = innerType;
  }

  private static String formatWithBrackets(String prefix, String content) {
    return prefix + "[" + content + "]";
  }

  private static String formatAsSelf(String typeName) {
    return formatWithBrackets("Self", typeName);
  }

  /**
   * Creates a SelfType wrapping the given type.
   * <p>
   * This factory method ensures the following invariants:
   * <ul>
   * <li>If the type is already a SelfType, it returns it unchanged</li>
   * <li>If the type is an ObjectType, it unwraps it and wraps the inner type in a SelfType, then wraps the result back in an ObjectType</li>
   * <li>If the type is a UnionType, it wraps each candidate in a SelfType and returns a new UnionType</li>
   * <li>Otherwise, it creates a new SelfType wrapping the given type</li>
   * </ul>
   *
   * @param type the type to wrap in a SelfType (can be null, in which case UNKNOWN is returned)
   * @return a PythonType representing Self[type], with proper structure maintained
   */
  public static PythonType of(@Nullable PythonType type) {
    if (type == null || type == PythonType.UNKNOWN) {
      return PythonType.UNKNOWN;
    }
    
    if (type instanceof SelfType) {
      return type;
    }
    
    if (type instanceof ObjectType objectType) {
      PythonType innerType = objectType.unwrappedType();
      SelfType selfType = new SelfType(innerType);
      return ObjectType.Builder.fromType(objectType)
        .withType(selfType)
        .build();
    }
    
    if (type instanceof UnionType unionType) {
      return UnionType.or(unionType.candidates().stream()
        .<PythonType>map(SelfType::new)
        .toList());
    }
    
    return new SelfType(type);
  }

  public PythonType innerType() {
    return innerType;
  }

  @Override
  public String name() {
    return formatAsSelf(innerType.name());
  }

  @Override
  public Optional<String> displayName() {
    return innerType.displayName()
      .map(SelfType::formatAsSelf);
  }

  @Override
  public Optional<String> instanceDisplayName() {
    return innerType.instanceDisplayName()
      .map(SelfType::formatAsSelf);
  }

  @Override
  public boolean isCompatibleWith(PythonType another) {
    return innerType.isCompatibleWith(another);
  }

  @Override
  public String key() {
    return formatAsSelf(innerType.key());
  }

  @Override
  public Optional<PythonType> resolveMember(String memberName) {
    return innerType.resolveMember(memberName);
  }

  @Override
  public TriBool hasMember(String memberName) {
    return innerType.hasMember(memberName);
  }

  @Override
  public Optional<LocationInFile> definitionLocation() {
    return innerType.definitionLocation();
  }

  @Override
  public PythonType unwrappedType() {
    return innerType.unwrappedType();
  }

  @Override
  public TypeSource typeSource() {
    return innerType.typeSource();
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    SelfType selfType = (SelfType) o;
    return Objects.equals(innerType, selfType.innerType);
  }

  @Override
  public int hashCode() {
    return Objects.hash(SelfType.class, innerType);
  }

  @Override
  public String toString() {
    return formatWithBrackets("SelfType", innerType.toString());
  }
}

