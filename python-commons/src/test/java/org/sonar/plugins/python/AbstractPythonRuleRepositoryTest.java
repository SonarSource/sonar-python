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

import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Test;
import org.sonar.api.SonarEdition;
import org.sonar.api.SonarQubeSide;
import org.sonar.api.SonarRuntime;
import org.sonar.api.batch.rule.CheckFactory;
import org.sonar.api.internal.SonarRuntimeImpl;
import org.sonar.api.server.rule.RulesDefinition;
import org.sonar.api.utils.Version;
import org.sonar.check.Rule;

import static org.assertj.core.api.Assertions.assertThat;

class AbstractPythonRuleRepositoryTest {
  private static final String DUMMY_KEY = "dummy";
  private static final String DUMMY_RESOURCE_FOLDER = "org/sonar/plugins/python/";
  private static final String DUMMY_LANGUAGE_KEY = "dummy-lang";

  @Test
  void testConstants() {
    var repo = createDummy(List.of(), Set.of(), Set.of());
    assertThat(repo.language()).isEqualTo(DUMMY_LANGUAGE_KEY);
  }

  @Test
  void registeringRules() {
    var repo = createDummy(List.of(DummyRuleCheck.class), Set.of(), Set.of());
    var rules = repo.rules();
    assertThat(rules).extracting(RulesDefinition.Rule::key).containsExactly("S9999");
    assertThat(rules).extracting(RulesDefinition.Rule::activatedByDefault).containsExactly(true);
  }

  @Test
  void templateRules() {
    var repo = createDummy(List.of(DummyRuleCheck.class), Set.of("S9999"), Set.of());
    var rules = repo.rules();
    assertThat(rules).extracting(RulesDefinition.Rule::template).containsExactly(true);
  }

  @Test
  void disabledRule() {
    var repo = createDummy(List.of(DummyRuleCheck.class), Set.of(), Set.of("S9999"));
    var rules = repo.rules();
    assertThat(rules).extracting(RulesDefinition.Rule::activatedByDefault).containsExactly(false);
  }

  @Rule(key = "S9999")
  private static class DummyRuleCheck extends PythonChecks {
    DummyRuleCheck(CheckFactory checkFactory) {
      super(checkFactory);
    }
  }

  private RulesDefinition.Repository createDummy(List<Class<?>> checks, Set<String> templateRuleKeys, Set<String> disabledRuleKeys) {
    var runtime = SonarRuntimeImpl.forSonarQube(Version.create(9, 3), SonarQubeSide.SERVER, SonarEdition.COMMUNITY);
    var dummyRepo = new DummyRuleRepository(runtime) {
      @Override
      protected List<Class<?>> getCheckClasses() {
        return checks;
      }

      @Override
      protected Set<String> getTemplateRuleKeys() {
        return templateRuleKeys;
      }

      @Override
      protected Set<String> getDisabledRules() {
        return disabledRuleKeys;
      }
    };
    var ctx = new RulesDefinition.Context();
    dummyRepo.define(ctx);
    return ctx.repository(DUMMY_KEY);
  }

  private static class DummyRuleRepository extends AbstractPythonRuleRepository {

    protected DummyRuleRepository(SonarRuntime sonarRuntime) {
      super(DUMMY_KEY, DUMMY_RESOURCE_FOLDER, DUMMY_LANGUAGE_KEY, sonarRuntime);
    }

    @Override
    protected List<Class<?>> getCheckClasses() {
      return List.of();
    }
  }

}
