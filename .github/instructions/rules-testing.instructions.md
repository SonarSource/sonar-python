---
applyTo: "**/*checks*/**/*.java"
---
Each rule implementation needs to be well tested.
To do so, we have powerful utilities in `PythonCheckVerifier` that allows for verification.

You can run the test for a given class with the shell command `mvn verify -DskipObfuscation -f"python-checks/pom.xml" -Dtest=$YOUR_TEST_CLASS`
Replace the $YOUR_TEST_CLASS by the name of the test class, usually of the form SleepZeroInAsyncCheckTest for the SleepZeroInAsyncCheck rule.

Look carefully at the terminal output, and make sure the test was successful.
If the build fails because of the licence headers, run `mvn license:format` from the root, and then re-try the test.

