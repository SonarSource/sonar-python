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

public class PythonRuleRepository extends AbstractPythonRuleRepository {

  public static final String REPOSITORY_KEY = "python";

  public PythonRuleRepository(SonarRuntime runtime) {
    super(REPOSITORY_KEY, OpenSourceCheckList.RESOURCE_FOLDER, Python.KEY, runtime);
  }

  protected List<Class<?>> getCheckClasses() {
    return new OpenSourceCheckList().getChecks().toList();
  }

  @Override
  protected Set<String> getTemplateRuleKeys() {
    return Collections.singleton("CommentRegularExpression");
  }
}
