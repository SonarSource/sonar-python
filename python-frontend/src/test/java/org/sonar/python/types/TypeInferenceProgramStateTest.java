/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2023 SonarSource SA
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
package org.sonar.python.types;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashSet;
import org.junit.jupiter.api.Test;
import org.sonar.python.cfg.fixpoint.ProgramState;
import org.sonar.python.semantic.SymbolImpl;

import static org.assertj.core.api.Assertions.assertThat;

public class TypeInferenceProgramStateTest {

  private final SymbolImpl a = new SymbolImpl("a", "foo.a");
  private final SymbolImpl b = new SymbolImpl("b", "foo.b");
  private final ProgramState otherProgramState = new ProgramState() {
    @Override
    public ProgramState join(ProgramState programState) {
      return null;
    }

    @Override
    public ProgramState copy() {
      return null;
    }
  };

  @Test
  public void test_equals() {
    TypeInferenceProgramState typeInferenceProgramState = new TypeInferenceProgramState();
    assertThat(typeInferenceProgramState)
      .isEqualTo(typeInferenceProgramState)
      .isEqualTo(new TypeInferenceProgramState())
      .isNotEqualTo(otherProgramState)
      .isNotEqualTo(null);

    typeInferenceProgramState.setTypes(a, Collections.singleton(InferredTypes.INT));
    typeInferenceProgramState.setTypes(b,  new HashSet<>(Arrays.asList(InferredTypes.BOOL, InferredTypes.STR)));
    TypeInferenceProgramState other = new TypeInferenceProgramState();
    other.setTypes(a, Collections.singleton(InferredTypes.INT));
    other.setTypes(b,  new HashSet<>(Arrays.asList(InferredTypes.BOOL, InferredTypes.STR)));
    assertThat(typeInferenceProgramState).isEqualTo(other);

    other = new TypeInferenceProgramState();
    other.setTypes(a, Collections.singleton(InferredTypes.INT));
    other.setTypes(b,  new HashSet<>(Arrays.asList(InferredTypes.INT, InferredTypes.STR)));
    assertThat(typeInferenceProgramState).isNotEqualTo(other);
  }

  @Test
  public void test_hashcode() {
    TypeInferenceProgramState typeInferenceProgramState = new TypeInferenceProgramState();
    typeInferenceProgramState.setTypes(a, Collections.singleton(InferredTypes.INT));
    typeInferenceProgramState.setTypes(b,  new HashSet<>(Arrays.asList(InferredTypes.BOOL, InferredTypes.STR)));

    TypeInferenceProgramState other = new TypeInferenceProgramState();
    other.setTypes(a, Collections.singleton(InferredTypes.INT));
    other.setTypes(b,  new HashSet<>(Arrays.asList(InferredTypes.BOOL, InferredTypes.STR)));

    assertThat(typeInferenceProgramState.hashCode()).isEqualTo(other.hashCode());
    assertThat(typeInferenceProgramState.hashCode()).isNotEqualTo(new TypeInferenceProgramState().hashCode());
  }

  @Test
  public void test_toString() {
    TypeInferenceProgramState typeInferenceProgramState = new TypeInferenceProgramState();
    typeInferenceProgramState.setTypes(a, Collections.singleton(InferredTypes.INT));
    typeInferenceProgramState.setTypes(b, new LinkedHashSet<>(Arrays.asList(InferredTypes.BOOL, InferredTypes.STR)));
    assertThat(typeInferenceProgramState.toString()).contains("b = RuntimeType(bool), RuntimeType(str)");
    assertThat(typeInferenceProgramState.toString()).contains("a = RuntimeType(int)");
  }
}
