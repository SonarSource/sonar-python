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
package org.sonar.python.semantic.v2;

import java.util.List;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.python.types.v2.Member;
import org.sonar.python.types.v2.ObjectType;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.TypeSource;
import org.sonar.python.types.v2.TypeWrapper;

public class ObjectTypeBuilder implements TypeBuilder<ObjectType> {

  private TypeWrapper typeWrapper;
  private List<PythonType> attributes;
  private List<Member> members;
  private TypeSource typeSource;

  @Override
  public ObjectType build() {
    return new ObjectType(typeWrapper, attributes, members, typeSource);
  }

  @Override
  public TypeBuilder<ObjectType> withDefinitionLocation(LocationInFile definitionLocation) {
    throw new IllegalStateException("Object type does not have definition location");
  }

  public ObjectTypeBuilder withTypeWrapper(TypeWrapper typeWrapper) {
    this.typeWrapper = typeWrapper;
    return this;
  }

  public ObjectTypeBuilder withAttributes(List<PythonType> attributes) {
    this.attributes = attributes;
    return this;
  }

  public ObjectTypeBuilder withMembers(List<Member> members) {
    this.members = members;
    return this;
  }

  public ObjectTypeBuilder withTypeSource(TypeSource typeSource) {
    this.typeSource = typeSource;
    return this;
  }

  public static ObjectTypeBuilder fromObjectType(ObjectType objectType) {
    return new ObjectTypeBuilder()
      .withTypeWrapper(objectType.typeWrapper())
      .withAttributes(objectType.attributes())
      .withMembers(objectType.members())
      .withTypeSource(objectType.typeSource());
  }
}
