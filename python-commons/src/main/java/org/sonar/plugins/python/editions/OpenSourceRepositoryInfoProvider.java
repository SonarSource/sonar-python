/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
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
package org.sonar.plugins.python.editions;

import java.util.Set;
import org.sonar.plugins.python.IPynbRuleRepository;
import org.sonar.plugins.python.PythonRuleRepository;
import org.sonar.python.checks.OpenSourceCheckList;

public class OpenSourceRepositoryInfoProvider implements RepositoryInfoProvider {
  @Override
  public RepositoryInfo getInfo() {
    return new RepositoryInfo(
      PythonRuleRepository.REPOSITORY_KEY,
      OpenSourceCheckList.SONAR_WAY_PROFILE_LOCATION,
      new OpenSourceCheckList().getChecks().toList(),
      Set.of()
    );
  }

  @Override
  public RepositoryInfo getIPynbInfo() {
    return new RepositoryInfo(
      IPynbRuleRepository.IPYTHON_REPOSITORY_KEY,
      OpenSourceCheckList.SONAR_WAY_PROFILE_LOCATION,
      new OpenSourceCheckList().getChecks().toList(),
      IPynbRuleRepository.DISABLED_RULES
    );
  }
}
