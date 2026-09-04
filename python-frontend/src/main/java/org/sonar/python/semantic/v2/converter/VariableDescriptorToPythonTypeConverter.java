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
import java.util.Set;
import org.sonar.plugins.python.api.types.v2.Member;
import org.sonar.plugins.python.api.types.v2.ObjectType;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.TypeWrapper;
import org.sonar.python.index.Descriptor;
import org.sonar.python.index.VariableDescriptor;
import org.sonar.python.types.v2.SpecialFormType;

public class VariableDescriptorToPythonTypeConverter implements DescriptorToPythonTypeConverter {
  private static final Set<String> SPECIAL_FORM_FQNS = Set.of("typing._SpecialForm", "typing_extensions._SpecialForm");

  public PythonType convert(ConversionContext ctx, VariableDescriptor from) {
    String fullyQualifiedName = from.fullyQualifiedName();
    if (from.isImportedModule() && fullyQualifiedName != null) {
      return ctx.lazyTypesContext().getOrCreateLazyType(fullyQualifiedName);
    }
    String annotatedType = from.annotatedType();
    if (annotatedType != null && from.isTypeAlias()) {
      // TypeAlias variables (e.g. typing.Text = str, socket.error = OSError) must resolve to
      // the aliased class itself, not to an instance of that class. We create a lazy type for
      // the target FQN so the v2 type system sees the class type (callable) rather than
      // ObjectType (an instance of the class, which would be non-callable).
      return ctx.lazyTypesContext().getOrCreateLazyType(annotatedType);
    }
    if (annotatedType != null) {
      if (SPECIAL_FORM_FQNS.contains(annotatedType) && fullyQualifiedName != null) {
        // Defensive null check on fullyQualifiedName: it should never be null for SpecialForm
        return new SpecialFormType(fullyQualifiedName);
      }
      TypeWrapper typeWrapper = ctx.lazyTypesContext().getOrCreateLazyTypeWrapper(annotatedType);

      List<PythonType> attributes = from.attributes().stream()
        .map(ctx::convert)
        .toList();

      List<Member> members = from.members().stream()
        .map(desc -> new Member(desc.name(), ctx.convert(desc)))
        .toList();

      return ObjectType.Builder.fromTypeWrapper(typeWrapper)
        .withAttributes(attributes)
        .withMembers(members)
        .build();
    }
    return PythonType.UNKNOWN;
  }

  @Override
  public PythonType convert(ConversionContext ctx, Descriptor from) {
    if (from instanceof VariableDescriptor variableDescriptor) {
      return convert(ctx, variableDescriptor);
    }
    throw new IllegalArgumentException("Unsupported Descriptor");
  }
}
