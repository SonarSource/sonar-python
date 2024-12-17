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
import java.util.ArrayList;
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
  record TestDuration(String test, Duration duration) {
  }

  private final static Logger LOGGER = LoggerFactory.getLogger(TestDurationMeasureExtension.class);

  private final Map<String, Instant> startTimeMap = new HashMap<>();
  private final List<TestDuration> durationList = new ArrayList<>();

  private void start(String uuid) {
    startTimeMap.put(uuid, Instant.now());
  }

  private void stop(String uuid, String name) {
    Instant start = startTimeMap.get(uuid);
    if (start != null) {
      Duration duration = Duration.between(start, Instant.now());
      durationList.add(new TestDuration(name, duration));
    }
  }

  @Override
  public void beforeEach(ExtensionContext context) {
    start(context.getUniqueId());
  }

  @Override
  public void afterEach(ExtensionContext context) {
    stop(context.getUniqueId(), context.getDisplayName());
  }

  @Override
  public void afterAll(ExtensionContext context) {
    LOGGER.info("Test durations:");
    durationList.sort(Comparator.comparing(TestDuration::duration).reversed());
    for (TestDuration duration : durationList) {
      LOGGER.info("{}: {} sec", duration.test(), duration.duration().toSeconds());
    }
  }
}
