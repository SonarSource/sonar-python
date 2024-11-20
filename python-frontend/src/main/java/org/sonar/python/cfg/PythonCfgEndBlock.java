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
package org.sonar.python.cfg;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import javax.annotation.CheckForNull;
import org.sonar.plugins.python.api.cfg.CfgBlock;
import org.sonar.plugins.python.api.tree.Tree;

public class PythonCfgEndBlock extends PythonCfgBlock {

  @Override
  public Set<CfgBlock> successors() {
    return Collections.emptySet();
  }

  @Override
  public List<Tree> elements() {
    return Collections.emptyList();
  }

  @Override
  void replaceSuccessors(Map<PythonCfgBlock, PythonCfgBlock> replacements) {
    // nothing to do
  }

  @CheckForNull
  @Override
  public CfgBlock syntacticSuccessor() {
    return null;
  }

  @Override
  public String toString() {
    return "END";
  }
}
