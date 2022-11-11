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
package org.sonar.python.caching;

import com.google.protobuf.InvalidProtocolBufferException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import org.sonar.plugins.python.api.caching.CacheContext;
import org.sonar.python.index.AmbiguousDescriptor;
import org.sonar.python.index.ClassDescriptor;
import org.sonar.python.index.Descriptor;
import org.sonar.python.index.DescriptorUtils;
import org.sonar.python.index.FunctionDescriptor;
import org.sonar.python.index.VariableDescriptor;
import org.sonar.python.types.protobuf.DescriptorsProtos;

public class Caching {

  private final CacheContext cacheContext;

  public static final String INPORT_MAP_CACHE_KEY_PREFIX = "python_import_map:";
  public static final String PROJECT_SYMBOL_TABLE_CACHE_KEY_PREFIX = "python_project_symbol_table:";

  public Caching(CacheContext cacheContext) {
    this.cacheContext = cacheContext;
  }

  public void writeImportMapEntry(String moduleFqn, Set<String> imports) {
    byte[] importData = String.join(";", imports).getBytes(StandardCharsets.UTF_8);
    String cacheKey = INPORT_MAP_CACHE_KEY_PREFIX + moduleFqn;
    cacheContext.getWriteCache().write(cacheKey, importData);
  }

  public void writeProjectLevelSymbolTableEntry(String moduleFqn, Set<Descriptor> descriptors) {
    String cacheKey = PROJECT_SYMBOL_TABLE_CACHE_KEY_PREFIX + moduleFqn;
    cacheContext.getWriteCache().write(cacheKey, moduleDescriptor(descriptors).toByteArray());
   }

  public Optional<Set<Descriptor>> readProjectLevelSymbolTableEntry(String moduleFqn) {
    cacheContext.getReadCache().contains(moduleFqn);
    byte[] bytes = cacheContext.getReadCache().readBytes(moduleFqn);
    if (bytes == null) {
      return Optional.empty();
    }
    try {
      return Optional.of(DescriptorUtils.deserializeProtobufDescriptors(bytes));
    } catch (InvalidProtocolBufferException e) {
      e.printStackTrace();
    }
    return Optional.empty();
  }

  public Optional<Set<String>> readImportMapEntry(String moduleFqn) {
    String cacheKey = INPORT_MAP_CACHE_KEY_PREFIX + moduleFqn;
    byte[] bytes = cacheContext.getReadCache().readBytes(cacheKey);
    if (bytes != null) {
      cacheContext.getWriteCache().copyFromPrevious(cacheKey);
      return Optional.of(new HashSet<>(Arrays.asList(new String(bytes, StandardCharsets.UTF_8).split(";"))));
    } else {
      return Optional.empty();
    }
  }

  public static DescriptorsProtos.ModuleDescriptor moduleDescriptor(Set<Descriptor> descriptors) {
    List<DescriptorsProtos.ClassDescriptor> classDescriptors = new ArrayList<>();
    List<DescriptorsProtos.FunctionDescriptor> functionDescriptors = new ArrayList<>();
    List<DescriptorsProtos.VarDescriptor> varDescriptors = new ArrayList<>();
    List<DescriptorsProtos.AmbiguousDescriptor> ambiguousDescriptors = new ArrayList<>();
    for (Descriptor descriptor : descriptors) {
      if (descriptor.kind().equals(Descriptor.Kind.CLASS)) {
        classDescriptors.add(((ClassDescriptor) descriptor).toProtobuf());
      } else if (descriptor.kind().equals(Descriptor.Kind.FUNCTION)) {
        functionDescriptors.add(((FunctionDescriptor) descriptor).toProtobuf());
      } else if (descriptor.kind().equals(Descriptor.Kind.VARIABLE)) {
        varDescriptors.add(((VariableDescriptor) descriptor).toProtobuf());
      } else {
        ambiguousDescriptors.add(((AmbiguousDescriptor) descriptor).toProtobuf());
      }
    }
    return DescriptorsProtos.ModuleDescriptor.newBuilder()
      .addAllClassDescriptors(classDescriptors)
      .addAllFunctionDescriptors(functionDescriptors)
      .addAllVarDescriptors(varDescriptors)
      .addAllAmbiguousDescriptors(ambiguousDescriptors)
      .build();
  }
}
