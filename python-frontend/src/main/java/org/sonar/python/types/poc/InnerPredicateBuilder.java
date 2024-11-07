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

import java.util.List;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.TriBool;
import org.sonar.python.types.v2.UnionType;

public interface InnerPredicateBuilder<I extends TypeCheckerBuilder<I>, O extends TypeCheckerBuilder<O>> {
  O construct(I input, TypeCheckerBuilderContext typeCheckerBuilderContext);

  default InnerPredicateBuilder<I, O> anyCandidate() {
    return (builder, ctx) -> {
      var wrappedContext = TypeCheckerBuilderContext.FakeTypeCheckBuilderContext.fromRealContext(ctx);
      var outputBuilder = this.construct(builder, wrappedContext);
      ctx.addPredicate(new AnyCandiateInnerPredicate(wrappedContext.getPredicates()));
      return outputBuilder.rebind(ctx);
    };
  }

  class AnyCandiateInnerPredicate implements InnerPredicate {

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
}
