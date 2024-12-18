/*
 * SonarQube Python Plugin
 * Copyright (C) 2012-2024 SonarSource SA
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
package org.sonar.python.it;

import java.time.Duration;
import java.time.Instant;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.extension.AfterAllCallback;
import org.junit.jupiter.api.extension.AfterEachCallback;
import org.junit.jupiter.api.extension.BeforeEachCallback;
import org.junit.jupiter.api.extension.ExtensionContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class TestDurationMeasureExtension implements BeforeEachCallback, AfterEachCallback, AfterAllCallback {
  private static final Logger LOGGER = LoggerFactory.getLogger(TestDurationMeasureExtension.class);

  private record TestEntry(String test, Duration duration, Instant start) {
    public TestEntry stoppedEntry() {
      if(duration.isZero()) {
        Duration newDuration = Duration.between(start, Instant.now());
        return new TestEntry(test, newDuration, start);
      } else {
        return this;
      }
    }
  }

  private final Map<String, TestEntry> testEntryMap = new HashMap<>();

  private synchronized void start(String uuid, String name) {
    testEntryMap.put(uuid, new TestEntry(name, Duration.ZERO, Instant.now()));
  }

  private synchronized void stop(String uuid) {
    testEntryMap.computeIfPresent(uuid, (key, entry) -> entry.stoppedEntry());
  }

  @Override
  public void beforeEach(ExtensionContext context) {
    start(context.getUniqueId(), context.getDisplayName());
  }

  @Override
  public void afterEach(ExtensionContext context) {
    stop(context.getUniqueId());
  }

  @Override
  public synchronized void afterAll(ExtensionContext context) {
    List<TestEntry> testEntries = testEntryMap.values().stream()
      .map(TestEntry::stoppedEntry)
      .sorted(Comparator.comparing(TestEntry::duration).reversed())
      .toList();

    LOGGER.info("Test durations:");
    for (TestEntry entry : testEntries) {
      LOGGER.info("{}: {} sec", entry.test(), entry.duration().toSeconds());
    }
  }
}
