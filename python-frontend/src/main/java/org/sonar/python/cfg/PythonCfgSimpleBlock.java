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
package org.sonar.python.cfg;

import java.util.Collections;
import java.util.Map;
import java.util.Set;
import javax.annotation.CheckForNull;
import org.sonar.plugins.python.api.cfg.CfgBlock;

public class PythonCfgSimpleBlock extends PythonCfgBlock {

  private CfgBlock successor;
  private CfgBlock syntacticSuccessor;


  public PythonCfgSimpleBlock(CfgBlock successor) {
    this.successor = successor;
  }

  @Override
  public Set<CfgBlock> successors() {
    return Collections.singleton(successor);
  }

  @CheckForNull
  @Override
  public CfgBlock syntacticSuccessor() {
    return syntacticSuccessor;
  }

  public void setSyntacticSuccessor(CfgBlock syntacticSuccessor) {
    this.syntacticSuccessor = syntacticSuccessor;
  }


  void replaceSuccessors(Map<PythonCfgBlock, PythonCfgBlock> replacements) {
    successor = replacements.getOrDefault(successor, (PythonCfgBlock) successor);
    if (syntacticSuccessor != null) {
      syntacticSuccessor = replacements.getOrDefault(syntacticSuccessor, (PythonCfgBlock) syntacticSuccessor);
    }
  }

}
