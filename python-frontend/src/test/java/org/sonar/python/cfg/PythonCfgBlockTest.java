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
package org.sonar.python.cfg;

import java.util.HashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class PythonCfgBlockTest {

  @Test
  void loop_of_empty_block() {
    PythonCfgEndBlock end = new PythonCfgEndBlock();
    PythonCfgSimpleBlock succ = new PythonCfgSimpleBlock(end);
    PythonCfgSimpleBlock block = new PythonCfgSimpleBlock(succ);
    PythonCfgSimpleBlock start = new PythonCfgSimpleBlock(block);
    Map<PythonCfgBlock, PythonCfgBlock> replacements = new HashMap<>();
    replacements.put(end, block);
    succ.replaceSuccessors(replacements);
    assertThat(start.firstNonEmptySuccessor()).isEqualTo(succ);
  }
}
