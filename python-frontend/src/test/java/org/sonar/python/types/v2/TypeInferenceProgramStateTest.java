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
package org.sonar.python.types.v2;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import org.junit.jupiter.api.Test;
import org.sonar.python.cfg.fixpoint.ProgramState;
import org.sonar.python.semantic.v2.SymbolV2;
import org.sonar.python.semantic.v2.types.TypeInferenceProgramState;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.types.v2.TypesTestUtils.BOOL_TYPE;
import static org.sonar.python.types.v2.TypesTestUtils.INT_TYPE;
import static org.sonar.python.types.v2.TypesTestUtils.STR_TYPE;

class TypeInferenceProgramStateTest {

  private final SymbolV2 a = new SymbolV2("a");
  private final SymbolV2 b = new SymbolV2("b");
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

    typeInferenceProgramState.setTypes(a, Collections.singleton(INT_TYPE));
    typeInferenceProgramState.setTypes(b,  new HashSet<>(Arrays.asList(BOOL_TYPE, STR_TYPE)));
    TypeInferenceProgramState other = new TypeInferenceProgramState();
    other.setTypes(a, Collections.singleton(INT_TYPE));
    other.setTypes(b,  new HashSet<>(Arrays.asList(BOOL_TYPE, STR_TYPE)));
    assertThat(typeInferenceProgramState).isEqualTo(other);

    other = new TypeInferenceProgramState();
    other.setTypes(a, Collections.singleton(INT_TYPE));
    other.setTypes(b,  new HashSet<>(Arrays.asList(INT_TYPE, STR_TYPE)));
    assertThat(typeInferenceProgramState).isNotEqualTo(other);
  }

  @Test
  void test_hashcode() {
    TypeInferenceProgramState typeInferenceProgramState = new TypeInferenceProgramState();
    typeInferenceProgramState.setTypes(a, Collections.singleton(INT_TYPE));
    typeInferenceProgramState.setTypes(b,  new HashSet<>(Arrays.asList(BOOL_TYPE, STR_TYPE)));

    TypeInferenceProgramState other = new TypeInferenceProgramState();
    other.setTypes(a, Collections.singleton(INT_TYPE));
    other.setTypes(b,  new HashSet<>(Arrays.asList(BOOL_TYPE, STR_TYPE)));

    assertThat(typeInferenceProgramState).hasSameHashCodeAs(other.hashCode());
    assertThat(typeInferenceProgramState.hashCode()).isNotEqualTo(new TypeInferenceProgramState().hashCode());
  }

}
