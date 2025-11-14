/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.plugins.python;

import java.util.Collections;
import java.util.List;
import java.util.Set;
import org.sonar.api.SonarRuntime;
import org.sonar.python.checks.OpenSourceCheckList;

public class IPynbRuleRepository extends AbstractPythonRuleRepository {

  public static final String IPYTHON_REPOSITORY_KEY = "ipython";
  public static final Set<String> DISABLED_RULES = Set.of("S905", "S2201", "S5754", "S1481");

  public IPynbRuleRepository(SonarRuntime runtime) {
    super(IPYTHON_REPOSITORY_KEY, OpenSourceCheckList.RESOURCE_FOLDER, IPynb.KEY, runtime);
  }

  @Override
  protected List<Class<?>> getCheckClasses() {
    return new OpenSourceCheckList().getChecks().toList();
  }

  @Override
  protected Set<String> getTemplateRuleKeys() {
    return Collections.singleton("CommentRegularExpression");
  }

  @Override
  protected Set<String> getDisabledRules() {
    return DISABLED_RULES;
  }
}
