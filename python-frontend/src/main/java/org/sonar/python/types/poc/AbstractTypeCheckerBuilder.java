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
package org.sonar.python.types.poc;

import java.util.Arrays;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Stream;
import org.sonar.python.types.poc.TypeCheckerBuilderContext.FakeTypeCheckBuilderContext;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.TriBool;

public abstract class AbstractTypeCheckerBuilder<SELF extends AbstractTypeCheckerBuilder<SELF>> implements TypeCheckerBuilder<SELF> {
  private final TypeCheckerBuilderContext context;

  protected AbstractTypeCheckerBuilder(TypeCheckerBuilderContext context) {
    this.context = context;
  }

  @Override
  public <O extends TypeCheckerBuilder<O>> O with(InnerPredicateBuilder<SELF, O> predicate) {
    // TODO fix generics (if possible)
    return predicate.construct((SELF) this, context);
  }

  @SafeVarargs
  @Override
  public final SELF or(Function<SELF, ? extends TypeCheckerBuilder<?>> firstPredicate,
    Function<SELF, ? extends TypeCheckerBuilder<?>>... otherPredicates) {
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

  private record OrInnerPredicate(List<List<InnerPredicate>> predicatesByOr) implements InnerPredicate {
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
