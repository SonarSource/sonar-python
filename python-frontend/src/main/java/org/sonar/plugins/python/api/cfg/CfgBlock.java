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
package org.sonar.plugins.python.api.cfg;

import java.util.List;
import java.util.Set;
import javax.annotation.CheckForNull;
import org.sonar.plugins.python.api.tree.Tree;

public interface CfgBlock {

  Set<CfgBlock> successors();

  Set<CfgBlock> predecessors();

  List<Tree> elements();

  /**
   * @return block following this one if no jump is applied
   * Returns {@code null} if this block doesn't end with jump statement (break, continue, return, raise)
   */
  @CheckForNull
  CfgBlock syntacticSuccessor();
}
