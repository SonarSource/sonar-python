/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
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
package org.sonar.python.types;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashSet;
import org.junit.jupiter.api.Test;
import org.sonar.python.cfg.fixpoint.ProgramState;
import org.sonar.python.semantic.SymbolImpl;

import static org.assertj.core.api.Assertions.assertThat;

class TypeInferenceProgramStateTest {

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
  void test_equals() {
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
  void test_hashcode() {
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
  void test_toString() {
    TypeInferenceProgramState typeInferenceProgramState = new TypeInferenceProgramState();
    typeInferenceProgramState.setTypes(a, Collections.singleton(InferredTypes.INT));
    typeInferenceProgramState.setTypes(b, new LinkedHashSet<>(Arrays.asList(InferredTypes.BOOL, InferredTypes.STR)));
    assertThat(typeInferenceProgramState.toString()).contains("b = RuntimeType(bool), RuntimeType(str)");
    assertThat(typeInferenceProgramState.toString()).contains("a = RuntimeType(int)");
  }
}
