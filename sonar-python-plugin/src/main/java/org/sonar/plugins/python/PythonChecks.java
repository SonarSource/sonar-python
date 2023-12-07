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
package org.sonar.plugins.python;

import com.google.common.collect.Streams;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;
import javax.annotation.Nullable;
import org.sonar.api.batch.rule.ActiveRules;
import org.sonar.api.batch.rule.CheckFactory;
import org.sonar.api.batch.rule.Checks;
import org.sonar.api.rule.RuleKey;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.plugins.python.api.PythonCustomRuleRepository;
import org.sonar.plugins.python.api.PythonSharedCheck;

public class PythonChecks {
  private final CheckFactory checkFactory;
  private final ActiveRules activeRules;
  private final List<Checks<PythonCheck>> checksByRepository = new ArrayList<>();
  private final List<PythonSharedCheck> sharedChecks = new ArrayList<>();

  PythonChecks(CheckFactory checkFactory, ActiveRules activeRules) {
    this.checkFactory = checkFactory;
    this.activeRules = activeRules;
  }

  public PythonChecks addChecks(String repositoryKey, Iterable<Class> checkClass) {
    var partition = StreamSupport
      .stream(checkClass.spliterator(), false)
      .collect(
        Collectors.partitioningBy(PythonChecks::isSharedCheckClass));

    var sharedCheckClasses = partition.get(true);
    var regularCheckClasses = partition.get(false);

    sharedCheckClasses.stream()
      .map(PythonChecks::instantiateSharedCheck)
      .filter(this::hasAtLeastOneActiveRule)
      .forEach(sharedChecks::add);

    checksByRepository.add(checkFactory.<PythonCheck>create(repositoryKey).addAnnotatedChecks(regularCheckClasses));

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
    return Streams.concat(
      checksByRepository.stream()
        .flatMap(c -> c.all().stream()),
      sharedChecks.stream())
      .collect(Collectors.toList());
  }

  @Nullable
  public RuleKey ruleKey(PythonCheck check) {
    return checksByRepository.stream().map(c -> c.ruleKey(check)).filter(Objects::nonNull).findFirst().orElse(null);
  }

  private static boolean isSharedCheckClass(Class checkClass) {
    return PythonSharedCheck.class.isAssignableFrom(checkClass);
  }

  private boolean hasAtLeastOneActiveRule(PythonSharedCheck sharedCheck) {
    return sharedCheck
      .ruleKeys()
      .stream()
      .anyMatch(ruleKey -> activeRules.find(ruleKey) != null);
  }

  private static PythonSharedCheck instantiateSharedCheck(Class sharedCheckClass) {
    try {
      @SuppressWarnings("unchecked")
      Class<? extends PythonSharedCheck> checkClassWithTypeAssumption = sharedCheckClass;

      return checkClassWithTypeAssumption
        .getDeclaredConstructor()
        .newInstance();
    } catch (InstantiationException | IllegalAccessException | InvocationTargetException | NoSuchMethodException | ClassCastException e) {
      throw new IllegalStateException(
        String.format("Failed to instantiate shared check %s for rules %s", sharedCheckClass, "TODO"),
        e);
    }
  }
}
