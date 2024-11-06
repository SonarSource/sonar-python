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
package org.sonar.python.types.v3;

import java.util.ArrayList;
import java.util.List;
import org.sonar.python.semantic.v2.ProjectLevelTypeTable;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.TriBool;

public class TypeCheckerPoc {

  // INTERFACES
  interface TypeCheckerBuilder<SELF extends TypeCheckerBuilder<SELF>> {
    <O extends TypeCheckerBuilder<O>> O with(InnerPredicateBuilder<SELF, O> predicate);

    TypeChecker build();
  }
  interface InnerPredicateBuilder<I extends TypeCheckerBuilder<? extends I>, O extends TypeCheckerBuilder<? extends O>> {
    O construct(I input, TypeCheckerBuilderContext typeCheckerBuilderContext);

    default InnerPredicateBuilder<I, O> anyCandidate() {
      return (builder, ctx) -> {
        return this.construct(builder, ctx);
      };
    }
  }

  interface InnerPredicate {
    TriBool apply(PythonType pythonType);
  }

  // CLASSES

  // TODO maybe move to interface
  static class TypeCheckerBuilderContext {
    private ProjectLevelTypeTable projectLevelTypeTable;

    TypeCheckerBuilderContext(ProjectLevelTypeTable projectLevelTypeTable) {
      this.projectLevelTypeTable = projectLevelTypeTable;
    }

    public ProjectLevelTypeTable getProjectLevelTypeTable() {
      return projectLevelTypeTable;
    }

  }

  abstract static class AbstractTypeCheckerBuilder<SELF extends AbstractTypeCheckerBuilder<SELF>> implements TypeCheckerBuilder<SELF> {
    private final TypeCheckerBuilderContext context;
    private List<InnerPredicate> predicates = new ArrayList<>();

    protected AbstractTypeCheckerBuilder(AbstractTypeCheckerBuilder<?> input) {
      this.predicates = new ArrayList<>(input.predicates);
      this.context = input.context;
    }

    private AbstractTypeCheckerBuilder(TypeCheckerBuilderContext context) {
      this.context = context;
    }

    public void addPredicate(InnerPredicate predicate) {
      predicates.add(predicate);
    }

    @Override
    public <O extends TypeCheckerBuilder<O>> O with(InnerPredicateBuilder<SELF, O> predicate) {
      // TODO fix generics (if possible)
      return predicate.construct((SELF) this, context);
    }

    public TypeChecker build() {
      return new TypeChecker(predicates);
    }
  }

  static class UnspecializedTypeCheckerBuilder extends AbstractTypeCheckerBuilder<UnspecializedTypeCheckerBuilder> {
    public UnspecializedTypeCheckerBuilder(TypeCheckerBuilderContext projectLevelTypeTable) {
      super(projectLevelTypeTable);
    }
  }

  static class ObjectTypeBuilder extends AbstractTypeCheckerBuilder<ObjectTypeBuilder> {
    public ObjectTypeBuilder(AbstractTypeCheckerBuilder<?> builder) {
      super(builder);
    }
  }

  static class ClassTypeBuilder extends AbstractTypeCheckerBuilder<ClassTypeBuilder> {
    public ClassTypeBuilder(AbstractTypeCheckerBuilder<?> builder) {
      super(builder);
    }
  }

  static class TypeChecker {
    private final List<InnerPredicate> predicates;

    public TypeChecker(List<InnerPredicate> predicates) {
      this.predicates = new ArrayList<>(predicates);
    }

    boolean isTrue(PythonType pythonType) {
      return predicates.stream().allMatch(predicate -> predicate.apply(pythonType) == TriBool.TRUE);
    }
  }
}
