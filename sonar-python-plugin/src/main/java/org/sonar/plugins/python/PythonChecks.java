/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2022 SonarSource SA
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
package org.sonar.plugins.python;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import javax.annotation.Nullable;
import org.sonar.api.batch.rule.CheckFactory;
import org.sonar.api.batch.rule.Checks;
import org.sonar.api.rule.RuleKey;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.plugins.python.api.PythonCustomRuleRepository;

public class PythonChecks {
  private final CheckFactory checkFactory;
  private List<Checks<PythonCheck>> checksByRepository = new ArrayList<>();

  PythonChecks(CheckFactory checkFactory) {
    this.checkFactory = checkFactory;
  }
  public PythonChecks addChecks(String repositoryKey, Iterable<Class> checkClass) {
    checksByRepository.add(checkFactory.<PythonCheck>create(repositoryKey).addAnnotatedChecks(checkClass));
    return this;
  }

  public PythonChecks addCustomChecks(@Nullable PythonCustomRuleRepository[] customRuleRepositories) {
    if (customRuleRepositories != null) {
      for (PythonCustomRuleRepository ruleRepository : customRuleRepositories) {
        addChecks(ruleRepository.repositoryKey(), ruleRepository.checkClasses());
      }
    }

    return this;
  }

  public List<PythonCheck> all() {
    return checksByRepository.stream().flatMap(c -> c.all().stream()).collect(Collectors.toList());
  }

  @Nullable
  public RuleKey ruleKey(PythonCheck check) {
    return checksByRepository.stream().map(c -> c.ruleKey(check)).filter(Objects::nonNull).findFirst().orElse(null);
  }

}
