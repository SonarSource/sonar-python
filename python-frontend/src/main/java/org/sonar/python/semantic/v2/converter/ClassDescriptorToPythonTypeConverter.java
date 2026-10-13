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
package org.sonar.python.semantic.v2.converter;

import java.util.List;
import org.sonar.plugins.python.api.types.v2.Member;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.SelfType;
import org.sonar.plugins.python.api.types.v2.TypeWrapper;
import org.sonar.python.index.ClassDescriptor;
import org.sonar.python.index.Descriptor;
import org.sonar.python.semantic.v2.ClassTypeBuilder;

public class ClassDescriptorToPythonTypeConverter implements DescriptorToPythonTypeConverter {

  private static PythonType convert(ConversionContext ctx, ClassDescriptor from) {
    var typeBuilder = new ClassTypeBuilder(from.name(), from.fullyQualifiedName())
      .withIsGeneric(from.supportsGenerics())
      .withHasDecorators(from.hasDecorators())
      .withDecorators(from.decorators().stream().map(decorator -> DecoratorDescriptorConverter.toType(ctx, decorator)).toList())
      .withDefinitionLocation(from.definitionLocation());

    from.superClasses().stream()
      .map(fqn -> {
        if (fqn != null) {
          return ctx.lazyTypesContext().getOrCreateLazyType(fqn);
        }
        return PythonType.UNKNOWN;
      })
      .map(TypeWrapper::of)
      .forEach(typeBuilder::addSuperClass);
    if (from.hasSuperClassWithoutDescriptor()) {
      typeBuilder.addSuperClass(PythonType.UNKNOWN);
    }

    List<PythonType> attributes = from.attributes().stream()
      .map(ctx::convert)
      .toList();
    typeBuilder.withAttributes(attributes);

    List<PythonType> metaClasses = from.metaClasses().stream()
      .map(ctx::convert)
      .toList();
    typeBuilder.withMetaClasses(metaClasses);

    var classType = typeBuilder.build();
    ctx.pushParent(classType);
    from.members()
      .stream()
      .map(d -> new Member(d.name(), ctx.convert(d)))
      .forEach(classType.members()::add);
    ctx.pollParent();

    if (from.isSelf()) {
      return SelfType.of(classType);
    }
    return classType;
  }

  @Override
  public PythonType convert(ConversionContext ctx, Descriptor from) {
    if (from instanceof ClassDescriptor classDescriptor) {
      return convert(ctx, classDescriptor);
    }
    throw new IllegalArgumentException("Unsupported Descriptor");
  }
}
