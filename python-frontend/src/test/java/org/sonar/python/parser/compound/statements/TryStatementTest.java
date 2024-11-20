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
package org.sonar.python.parser.compound.statements;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.parser.RuleTest;

import static org.sonar.python.parser.PythonParserAssert.assertThat;

class TryStatementTest extends RuleTest {

  @BeforeEach
  void init() {
    setRootRule(PythonGrammar.TRY_STMT);
  }

  @Test
  void ok() {

    assertThat(p).matches("try : pass\nexcept e : pass")
      .matches("try : pass\nexcept e : pass\nexcept f : pass")
      .matches("try : pass\nexcept e : pass\nelse : pass")
      .matches("try : pass\nexcept e : pass\nfinally : pass")
      .matches("try : pass\nexcept e : pass\nelse : pass\nfinally : pass")
      .matches("try : pass\nfinally : pass")
      .matches("try : pass\nexcept* e : pass");
  }

}
