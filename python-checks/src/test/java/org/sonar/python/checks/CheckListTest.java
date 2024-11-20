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
package org.sonar.python.checks;

import com.google.common.collect.Iterables;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.apache.commons.io.FileUtils;
import org.junit.jupiter.api.Test;
import org.sonarsource.analyzer.commons.internal.json.simple.JSONArray;
import org.sonarsource.analyzer.commons.internal.json.simple.JSONObject;
import org.sonarsource.analyzer.commons.internal.json.simple.parser.JSONParser;
import org.sonarsource.analyzer.commons.internal.json.simple.parser.ParseException;

import static java.nio.charset.StandardCharsets.UTF_8;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.fail;

class CheckListTest {

  private static final Path METADATA_DIR = Paths.get("src/main/resources/org/sonar/l10n/py/rules/python");

  private static final Pattern SQ_KEY = Pattern.compile("\"sqKey\": \"([^\"]*)\"");

  /**
   * Enforces that each check declared in list.
   */
  @Test
  void count() {
    int count = 0;
    List<File> files = (List<File>) FileUtils.listFiles(new File("src/main/java/org/sonar/python/checks/"), new String[]{"java"}, true);
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
  void test() {
    Iterable<Class<?>> checks = CheckList.getChecks();

    for (Class<?> cls : checks) {
      String testName = '/' + cls.getName().replace('.', '/') + "Test.class";
      assertThat(getClass().getResource(testName))
        .overridingErrorMessage("No test for " + cls.getSimpleName())
        .isNotNull();
    }
  }

  @Test
  void validate_sqKey_field_in_json() throws IOException {
    try (Stream<Path> fileStream = Files.find(METADATA_DIR, 1, (path, attr) -> path.toString().endsWith(".json"))) {
      List<Path> jsonList = fileStream
        .filter(path -> !path.toString().endsWith("Sonar_way_profile.json"))
        .sorted()
        .toList();

      List<String> fileNames = jsonList.stream()
        .map(Path::getFileName)
        .map(Path::toString)
        .map(name -> name.replaceAll("\\.json$", ""))
        .toList();

      List<String> sqKeys = jsonList.stream()
        .map(CheckListTest::extractSqKey)
        .toList();

      assertThat(fileNames).isEqualTo(sqKeys);
    }
  }

  @Test
  void test_no_deprecated_rule_in_default_profile() throws IOException, ParseException {
    try (Stream<Path> fileStream = Files.find(METADATA_DIR, 1, (path, attr) -> path.toString().endsWith(".json"))) {
      List<Path> jsonList = fileStream.toList();

      Path sonarWayProfilePath = jsonList.stream().filter(path -> path.toString().endsWith("Sonar_way_profile.json")).findFirst().get();
      List<String> keysInDefaultProfile = getKeysInDefaultProfile(sonarWayProfilePath);

      Set<String> deprecatedKeys = jsonList.stream()
        .filter(path -> !path.toString().endsWith("Sonar_way_profile.json"))
        .filter(path1 -> {
          try {
            return isDeprecated(path1);
          } catch (Exception e) {
            fail(String.format("Exception when deserializing JSON file \"%s\"", path1.getFileName().toString()));
            return false;
          }
        })
        .map(Path::getFileName)
        .map(Path::toString)
        .map(name -> name.replaceAll("\\.json$", ""))
        .collect(Collectors.toSet());

      assertThat(keysInDefaultProfile).isNotEmpty();
      assertThat(deprecatedKeys).isNotEmpty();
      assertThat(keysInDefaultProfile).doesNotContainAnyElementsOf(deprecatedKeys);
    }
  }

  @Test
  void test_locally_deprecated_rules_stay_deprecated() throws IOException, ParseException {
    // Some rules have been deprecated only for Python. When executed, rule-api reverts those rule to "ready" status, which is incorrect.
    // This test is here to ensure it doesn't happen.
    List<String> locallyDeprecatedRules = Arrays.asList("S1523", "S4721");
    try (Stream<Path> fileStream = Files.find(METADATA_DIR, 1, (path, attr) -> path.toString().endsWith(".json"))) {
      Set<String> deprecatedKeys = fileStream
        .filter(path -> !path.toString().endsWith("Sonar_way_profile.json"))
        .filter(path1 -> {
          try {
            return isDeprecated(path1);
          } catch (Exception e) {
            fail(String.format("Exception when deserializing JSON file \"%s\"", path1.getFileName().toString()));
            return false;
          }
        })
        .map(Path::getFileName)
        .map(Path::toString)
        .map(name -> name.replaceAll("\\.json$", ""))
        .collect(Collectors.toSet());

      assertThat(deprecatedKeys).containsAll(locallyDeprecatedRules);
    }
  }

  private static boolean isDeprecated(Path path) throws IOException, ParseException {
    InputStream in = new FileInputStream(path.toFile());
    JSONParser jsonParser = new JSONParser();
    JSONObject ruleJson = (JSONObject) jsonParser.parse(new InputStreamReader(in, UTF_8));
    Object status = ruleJson.get("status");
    return status.equals("deprecated");
  }

  private static List<String> getKeysInDefaultProfile(Path sonarWayPath) throws IOException, ParseException {
    InputStream in = new FileInputStream(sonarWayPath.toFile());
    JSONParser jsonParser = new JSONParser();
    JSONObject sonarWayJson = (JSONObject) jsonParser.parse(new InputStreamReader(in, UTF_8));
    JSONArray sonarWayKeys = (JSONArray) sonarWayJson.get("ruleKeys");
    return (List<String>) sonarWayKeys.stream().sorted().toList();
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
