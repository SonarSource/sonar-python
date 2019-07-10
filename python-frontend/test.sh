#!/bin/bash
java -verbose:class -cp target/classes:target/test-classes:\
lib/extensions.jar:\
lib/openapi.jar:\
lib/platform-api.jar:\
lib/platform-impl.jar:\
lib/pycharm.jar:\
lib/pycharm-pydev.jar:\
lib/resources_en.jar:\
lib/util.jar:\
lib/trove4j.jar:\
lib/picocontainer-1.2.jar:\
lib/kotlin-stdlib-1.3.11.jar:\
lib/guava-25.1-jre.jar\
 org.sonar.python.frontend.MyTest