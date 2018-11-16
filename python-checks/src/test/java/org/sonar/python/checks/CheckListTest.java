/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2018 SonarSource SA
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
package org.sonar.python.checks;

import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collection;
import java.util.List;
import java.util.Locale;
import java.util.ResourceBundle;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import org.apache.commons.io.FileUtils;
import org.junit.Test;
import org.sonar.api.batch.rule.ActiveRules;
import org.sonar.api.batch.rule.CheckFactory;
import org.sonar.api.batch.rule.internal.ActiveRulesBuilder;
import org.sonar.api.rules.Rule;
import org.sonar.api.rules.RuleParam;

import static java.nio.charset.StandardCharsets.UTF_8;
import static org.assertj.core.api.Assertions.assertThat;

public class CheckListTest {

  private static final Path METADATA_DIR = Paths.get("src/main/resources/org/sonar/l10n/py/rules/python");

  private static final Pattern SQ_KEY = Pattern.compile("\"sqKey\": \"([^\"]*)\"");

  /**
   * Enforces that each check declared in list.
   */
  @Test
  public void count() {
    int count = 0;
    List<File> files = (List<File>) FileUtils.listFiles(new File("src/main/java/org/sonar/python/checks/"), new String[] {"java"}, false);
    for (File file : files) {
      if (file.getName().endsWith("Check.java") && !file.getName().startsWith("Abstract")) {
        count++;
      }
    }
    assertThat(Iterables.size(CheckList.getChecks())).isEqualTo(count);
  }

  /**
   * Enforces that each check has test, name and description.
   */
  @Test
  public void test() {
    Iterable<Class> checks = CheckList.getChecks();

    for (Class cls : checks) {
      String testName = '/' + cls.getName().replace('.', '/') + "Test.class";
      assertThat(getClass().getResource(testName))
          .overridingErrorMessage("No test for " + cls.getSimpleName())
          .isNotNull();
    }

    ResourceBundle resourceBundle = ResourceBundle.getBundle("org.sonar.l10n.py", Locale.ENGLISH);

    Set<String> keys = Sets.newHashSet();

    ActiveRules activeRules = (new ActiveRulesBuilder())
        .build();
    CheckFactory checkFactory = new CheckFactory(activeRules);
    Collection<Rule> rules = checkFactory
        .<Rule>create("repositoryKey")
        .addAnnotatedChecks(CheckList.getChecks())
        .all();
    for (Rule rule : rules) {
      assertThat(keys).as("Duplicate key " + rule.getKey()).doesNotContain(rule.getKey());
      keys.add(rule.getKey());

      resourceBundle.getString("rule." + CheckList.REPOSITORY_KEY + "." + rule.getKey() + ".name");
      assertThat(getClass().getResource("/org/sonar/l10n/python/rules/python/" + rule.getKey() + ".html"))
          .overridingErrorMessage("No description for " + rule.getKey())
          .isNotNull();

      assertThat(rule.getDescription())
          .overridingErrorMessage("Description of " + rule.getKey() + " should be in separate file")
          .isNull();

      for (RuleParam param : rule.getParams()) {
        resourceBundle.getString("rule." + CheckList.REPOSITORY_KEY + "." + rule.getKey() + ".param." + param.getKey());

        assertThat(param.getDescription())
            .overridingErrorMessage("Description for param " + param.getKey() + " of " + rule.getKey() + " should be in separate file")
            .isEmpty();
      }
    }
  }

  @Test
  public void validate_sqKey_field_in_json() throws IOException {
    List<Path> jsonList = Files.find(METADATA_DIR, 1, (path, attr) -> path.toString().endsWith(".json"))
      .filter(path -> !path.toString().endsWith("Sonar_way_profile.json"))
      .sorted()
      .collect(Collectors.toList());

    List<String> fileNames = jsonList.stream()
      .map(Path::getFileName)
      .map(Path::toString)
      .map(name -> name.replaceAll("\\.json$", ""))
      .collect(Collectors.toList());

    List<String> sqKeys = jsonList.stream()
      .map(CheckListTest::extractSqKey)
      .collect(Collectors.toList());

    assertThat(fileNames).isEqualTo(sqKeys);
  }

  private static String extractSqKey(Path jsonFile) {
    try {
      String content = new String(Files.readAllBytes(jsonFile), UTF_8);
      Matcher matcher = SQ_KEY.matcher(content);
      if (!matcher.find()) {
        return "Can not find sqKey in " + jsonFile;
      }
      return matcher.group(1);
    } catch (IOException e) {
      throw new IllegalStateException(e);
    }
  }
}
