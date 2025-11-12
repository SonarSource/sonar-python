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

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class RepositoryInfoProviderWrapperTest {

  @Test
  void testEmptyConstructor() {
    RepositoryInfoProviderWrapper repositoryInfoProviderWrapper = new RepositoryInfoProviderWrapper();
    assertThat(repositoryInfoProviderWrapper.infoProviders()).hasSize(1);
    assertThat(repositoryInfoProviderWrapper.infoProviders()[0]).isInstanceOf(OpenSourceRepositoryInfoProvider.class);
  }

  @Test
  void testConstructorWithParameter() {
    RepositoryInfoProvider[] editionMetadataProviders = new RepositoryInfoProvider[] {new OpenSourceRepositoryInfoProvider(), new OpenSourceRepositoryInfoProvider()};
    RepositoryInfoProviderWrapper repositoryInfoProviderWrapper = new RepositoryInfoProviderWrapper(editionMetadataProviders);
    assertThat(repositoryInfoProviderWrapper.infoProviders()).hasSize(2);
    assertThat(repositoryInfoProviderWrapper.infoProviders()[0]).isInstanceOf(OpenSourceRepositoryInfoProvider.class);
    assertThat(repositoryInfoProviderWrapper.infoProviders()[1]).isInstanceOf(OpenSourceRepositoryInfoProvider.class);
  }

}
