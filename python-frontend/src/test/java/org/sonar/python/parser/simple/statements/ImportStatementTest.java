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
package org.sonar.python.parser.simple.statements;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.parser.RuleTest;

import static org.sonar.python.parser.PythonParserAssert.assertThat;

class ImportStatementTest extends RuleTest {

  @BeforeEach
  void init() {
    setRootRule(PythonGrammar.IMPORT_STMT);
  }

  @Test
  void realLife() {
    assertThat(p).matches("from boto.s3.connection import S3Connection");
    assertThat(p).matches("import sys");
  }

  @Test
  void lazyImports() {
    setRootRule(PythonGrammar.FILE_INPUT);

    assertThat(p).matches("lazy import json\n");
    assertThat(p).matches("lazy from pathlib import Path\n");
    assertThat(p).matches("lazy from ..package import name as alias\n");
    assertThat(p).matches("lazy = 42\nlazy()\n");
    assertThat(p).matches("if TYPE_CHECKING:\n  lazy import json\n");
    // Ordinary module references, not imports from the special absolute __future__ module.
    assertThat(p).matches("lazy from __future__.something import name\n");
    assertThat(p).matches("lazy from __future__.something import (name as alias)\n");
    assertThat(p).matches("lazy from .__future__ import annotations\n");
    assertThat(p).matches("lazy from __future__x import name\n");
    assertThat(p).matches("import json; lazy import pathlib\n");
    assertThat(p).matches("lazy import json; lazy import pathlib\n");
    // Lazy star imports and absolute future imports are invalid Python syntax,
    // intentionally parsed so that analysis can continue.
    assertThat(p).matches("lazy from package import *\n");
    assertThat(p).matches("lazy from __future__ import annotations\n");
    assertThat(p).matches("lazy from __future__ import (annotations as alias)\n");
  }

  @Test
  void lazyImportsInIPython() {
    setRootRule(PythonGrammar.FILE_INPUT);

    assertThat(ip).matches("lazy import json\n");
    assertThat(ip).matches("lazy from pathlib import Path\n");
    assertThat(ip).matches("lazy = 42\nlazy()\n");
  }

}
