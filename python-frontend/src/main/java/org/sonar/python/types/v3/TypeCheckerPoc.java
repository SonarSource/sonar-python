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
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Stream;
import org.sonar.python.semantic.v2.ProjectLevelTypeTable;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.TriBool;
import org.sonar.python.types.v2.UnionType;

public class TypeCheckerPoc {

  // INTERFACES
  interface TypeCheckerBuilder<SELF extends TypeCheckerBuilder<SELF>> {
    <O extends TypeCheckerBuilder<O>> O with(InnerPredicateBuilder<SELF, O> predicate);

    SELF or(Function<SELF, ? extends TypeCheckerBuilder<?>> firstPredicate, Function<SELF, ? extends TypeCheckerBuilder<?>>... otherPredicates);

    TypeChecker build();

    SELF rebind(TypeCheckerBuilderContext context);

  }
  interface InnerPredicateBuilder<I extends TypeCheckerBuilder<I>, O extends TypeCheckerBuilder<O>> {
    O construct(I input, TypeCheckerBuilderContext typeCheckerBuilderContext);

    default InnerPredicateBuilder<I, O> anyCandidate() {
      return (builder, ctx) -> {
        var wrappedContext = FakeTypeCheckBuilderContext.fromRealContext(ctx);
        var outputBuilder = this.construct(builder, wrappedContext);
        ctx.addPredicate(new AnyCandiateInnerPredicate(wrappedContext.getPredicates()));
        return outputBuilder.rebind(ctx);
      };
    }
  }

  private static class AnyCandiateInnerPredicate implements InnerPredicate {

    private final List<InnerPredicate> predicates;

    public AnyCandiateInnerPredicate(List<InnerPredicate> predicates) {
      this.predicates = predicates;
    }

    @Override
    public TriBool apply(PythonType pythonType) {
      if (pythonType instanceof UnionType unionType) {
        TriBool latestResut = TriBool.FALSE;
        for (PythonType candidate : unionType.candidates()) {
          var result = applyPredicates(candidate);
          if (result == TriBool.TRUE) {
            return TriBool.TRUE;
          } else if (result == TriBool.UNKNOWN) {
            latestResut = TriBool.UNKNOWN;
          }
        }
        return latestResut;
      } else {
        return applyPredicates(pythonType);
      }
    }

    private TriBool applyPredicates(PythonType pythonType) {
      TriBool result = TriBool.TRUE;
      for (InnerPredicate predicate : predicates) {
        result = result.and(predicate.apply(pythonType));
      }
      return result;
    }
  }

  interface InnerPredicate {
    TriBool apply(PythonType pythonType);
  }

  // CLASSES

  // TODO maybe move to interface
  static class TypeCheckerBuilderContext {
    private final ProjectLevelTypeTable projectLevelTypeTable;
    protected final List<InnerPredicate> predicates = new ArrayList<>();

    TypeCheckerBuilderContext(ProjectLevelTypeTable projectLevelTypeTable) {
      this.projectLevelTypeTable = projectLevelTypeTable;
    }

    public ProjectLevelTypeTable getProjectLevelTypeTable() {
      return projectLevelTypeTable;
    }

    public void addPredicate(InnerPredicate predicate) {
      predicates.add(predicate);
    }
  }

  static class FakeTypeCheckBuilderContext extends TypeCheckerBuilderContext {

    private FakeTypeCheckBuilderContext(ProjectLevelTypeTable projectLevelTypeTable) {
      super(projectLevelTypeTable);
    }

    public List<InnerPredicate> getPredicates() {
      return predicates;
    }

    static FakeTypeCheckBuilderContext fromRealContext(TypeCheckerBuilderContext ctx) {
      return new FakeTypeCheckBuilderContext(ctx.getProjectLevelTypeTable());
    }
  }

  abstract static class AbstractTypeCheckerBuilder<SELF extends AbstractTypeCheckerBuilder<SELF>> implements TypeCheckerBuilder<SELF> {
    private TypeCheckerBuilderContext context;

    private AbstractTypeCheckerBuilder(TypeCheckerBuilderContext context) {
      this.context = context;
    }

    @Override
    public <O extends TypeCheckerBuilder<O>> O with(InnerPredicateBuilder<SELF, O> predicate) {
      // TODO fix generics (if possible)
      return predicate.construct((SELF) this, context);
    }

    @SafeVarargs
    @Override
    public final SELF or(Function<SELF, ? extends TypeCheckerBuilder<?>> firstPredicate, Function<SELF, ? extends TypeCheckerBuilder<?>>... otherPredicates) {
      // TODO (maybe) refactor the List<List<...>>
      List<List<InnerPredicate>> predicatesByOr = Stream.concat(Stream.of(firstPredicate), Arrays.stream(otherPredicates))
        .map(builder -> {
          var newCtx = FakeTypeCheckBuilderContext.fromRealContext(context);
          builder.apply(rebind(newCtx)).build();
          return newCtx.getPredicates();
        }).toList();

      context.addPredicate(new OrInnerPredicate(predicatesByOr));

      // TODO fix generics (if possible)
      return (SELF) this;
    }

    private static class OrInnerPredicate implements InnerPredicate {
      private final List<List<InnerPredicate>> predicatesByOr;

      public OrInnerPredicate(List<List<InnerPredicate>> predicatesByOr) {
        this.predicatesByOr = predicatesByOr;
      }

      @Override
      public TriBool apply(PythonType type) {
        var result = TriBool.FALSE;
        // Logic should be checked again on implementation
        for (List<InnerPredicate> predicates : predicatesByOr) {
          var currentResult = TriBool.TRUE;
          for (InnerPredicate predicate : predicates) {
            currentResult = currentResult.and(predicate.apply(type));
          }
          if (currentResult == TriBool.TRUE) {
            return TriBool.TRUE;
          } else if (currentResult == TriBool.UNKNOWN) {
            result = TriBool.UNKNOWN;
          }

        }
        return result;
      }
    }


    @Override
    public TypeChecker build() {
      if (context.predicates.isEmpty()) {
        throw new IllegalStateException("No predicates were added");
      }
      return new TypeChecker(context.predicates);
    }
  }

  static class UnspecializedTypeCheckerBuilder extends AbstractTypeCheckerBuilder<UnspecializedTypeCheckerBuilder> {
    public UnspecializedTypeCheckerBuilder(TypeCheckerBuilderContext projectLevelTypeTable) {
      super(projectLevelTypeTable);
    }

    @Override
    public UnspecializedTypeCheckerBuilder rebind(TypeCheckerBuilderContext context) {
      return new UnspecializedTypeCheckerBuilder(context);
    }
  }

  static class ObjectTypeBuilder extends AbstractTypeCheckerBuilder<ObjectTypeBuilder> {
    public ObjectTypeBuilder(TypeCheckerBuilderContext context) {
      super(context);
    }

    @Override
    public ObjectTypeBuilder rebind(TypeCheckerBuilderContext context) {
      return new ObjectTypeBuilder(context);
    }

  }

  static class ClassTypeBuilder extends AbstractTypeCheckerBuilder<ClassTypeBuilder> {
    public ClassTypeBuilder(TypeCheckerBuilderContext context) {
      super(context);
    }

    @Override
    public ClassTypeBuilder rebind(TypeCheckerBuilderContext context) {
      return new ClassTypeBuilder(context);
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
