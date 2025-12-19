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
  private final TypeWrapper typeWrapper;

  private SelfType(TypeWrapper typeWrapper) {
    this.typeWrapper = typeWrapper;
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
   * <li>If the type is null, UNKNOWN, or not a supported type, it returns UNKNOWN</li>
   * <li>If the type is already a SelfType, it returns it unchanged</li>
   * <li>The type <code>ObjectType[ClassType[A]]</code> is converted to <code>ObjectType[SelfType[ClassType[A]]]</code></li>
   * <li>The type ObjectType[UnionType[ClassType[A], ClassType[B]]] is converted to ObjectType[UnionType[SelfType[ClassType[A]], SelfType[ClassType[B]]]]</li>
   * <li>The type ClassType[A] is converted to SelfType[ClassType[A]]</li>
   * </ul>
   *
   * @param type the type to wrap in a SelfType (can be null, in which case UNKNOWN is returned)
   * @return a PythonType representing Self[type], with proper structure maintained, or UNKNOWN if the type is not supported
   */
  public static PythonType of(@Nullable PythonType type) {
    if (type == null || type == PythonType.UNKNOWN) {
      return PythonType.UNKNOWN;
    }

    if (type instanceof SelfType) {
      return type;
    }

    if (type instanceof ClassType classType) {
      return new SelfType(TypeWrapper.of(classType));
    }

    if (type instanceof UnionType unionType) {
      return UnionType.or(unionType.candidates().stream()
        .map(SelfType::of)
        .toList());
    }

    if (type instanceof ObjectType objectType) {
      PythonType unwrapped = objectType.unwrappedType();
      PythonType wrappedType = of(unwrapped);

      if (wrappedType != PythonType.UNKNOWN) {
        return ObjectType.Builder.fromType(objectType)
          .withType(wrappedType)
          .build();
      }
      return PythonType.UNKNOWN;
    }
    return PythonType.UNKNOWN;
  }

  public static PythonType fromTypeWrapper(TypeWrapper typeWrapper){
    return new SelfType(typeWrapper);
  }

  public PythonType innerType() {
    var type = typeWrapper.type();
    if (type instanceof ClassType) {
      return type;
    }
    return PythonType.UNKNOWN;
  }

  public TypeWrapper typeWrapper() {
    return typeWrapper;
  }

  @Override
  public String name() {
    return formatAsSelf(typeWrapper.type().name());
  }

  @Override
  public Optional<String> displayName() {
    return typeWrapper.type().displayName()
      .map(SelfType::formatAsSelf);
  }

  @Override
  public Optional<String> instanceDisplayName() {
    return typeWrapper.type().instanceDisplayName()
      .map(SelfType::formatAsSelf);
  }

  @Override
  public TriBool isCompatibleWith(PythonType another) {
    return typeWrapper.type().isCompatibleWith(another);
  }

  @Override
  public String key() {
    return formatAsSelf(typeWrapper.type().key());
  }

  @Override
  public Optional<PythonType> resolveMember(String memberName) {
    return typeWrapper.type().resolveMember(memberName);
  }

  @Override
  public TriBool hasMember(String memberName) {
    return typeWrapper.type().hasMember(memberName);
  }

  @Override
  public Optional<LocationInFile> definitionLocation() {
    return typeWrapper.type().definitionLocation();
  }

  @Override
  public PythonType unwrappedType() {
    return typeWrapper.type().unwrappedType();
  }

  @Override
  public TypeSource typeSource() {
    return typeWrapper.type().typeSource();
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    SelfType selfType = (SelfType) o;
    return Objects.equals(typeWrapper, selfType.typeWrapper);
  }

  @Override
  public int hashCode() {
    return Objects.hash(SelfType.class, typeWrapper);
  }

  @Override
  public String toString() {
    return formatWithBrackets("SelfType", typeWrapper.toString());
  }
}
