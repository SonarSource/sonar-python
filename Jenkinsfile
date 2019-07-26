@Library('SonarSource@2.2') _
pipeline {
  agent {
    label 'linux'
  }
  parameters {
    string(name: 'GIT_SHA1', description: 'Git SHA1 (provided by travisci hook job)')
    string(name: 'CI_BUILD_NAME', defaultValue: 'sonar-python', description: 'Build Name (provided by travisci hook job)')
    string(name: 'CI_BUILD_NUMBER', description: 'Build Number (provided by travisci hook job)')
    string(name: 'GITHUB_BRANCH', defaultValue: 'master', description: 'Git branch (provided by travisci hook job)')
    string(name: 'GITHUB_REPOSITORY_OWNER', defaultValue: 'SonarSource', description: 'Github repository owner(provided by travisci hook job)')
  }
  environment {
    SONARSOURCE_QA = 'true'
    MAVEN_TOOL = 'Maven 3.6.x'
    JDK_VERSION = 'Java 11'
    PYCHARM_VERSION = '2019.1.3'
  }
  stages {
    stage('Notify') {
      steps {
        sendAllNotificationQaStarted()
      }
    }
    stage('QA') {
      parallel {
        stage('plugin/DOGFOOD/linux') {
          agent {
            label 'linux'
          }
          steps {
            runITs("plugin","DOGFOOD")
          }
        }     
        stage('plugin/LATEST_RELEASE[6.7]/linux') {
          agent {
            label 'linux'
          }
          steps {
            runITs("plugin","LATEST_RELEASE[6.7]")
          }
        }
        stage('ruling/LATEST_RELEASE/linux') {
          agent {
            label 'linux'
          }
          steps {
            runITs("ruling","LATEST_RELEASE")            
          }
        }
        stage('ci/windows') {
          agent {
            label 'windows'
          }
          steps {
            withMaven(maven: MAVEN_TOOL) {
              mavenSetBuildVersion()
              sh "curl -L -O https://download-cf.jetbrains.com/python/pycharm-community-${PYCHARM_VERSION}.tar.gz"
              sh "tar xzf pycharm-community-${PYCHARM_VERSION}.tar.gz"
              sh "rm pycharm-community-${PYCHARM_VERSION}.tar.gz"
              dir("pycharm-community-$PYCHARM_VERSION/lib") {
                runMaven(JDK_VERSION,"install:install-file -Dfile=extensions.jar -DgroupId=com.jetbrains.pycharm -DartifactId=extensions -Dversion=${PYCHARM_VERSION} -Dpackaging=jar")
                runMaven(JDK_VERSION,"install:install-file -Dfile=openapi.jar -DgroupId=com.jetbrains.pycharm -DartifactId=openapi -Dversion=${PYCHARM_VERSION} -Dpackaging=jar")
                runMaven(JDK_VERSION,"install:install-file -Dfile=platform-api.jar -DgroupId=com.jetbrains.pycharm -DartifactId=platform-api -Dversion=${PYCHARM_VERSION} -Dpackaging=jar")
                runMaven(JDK_VERSION,"install:install-file -Dfile=platform-impl.jar -DgroupId=com.jetbrains.pycharm -DartifactId=platform-impl -Dversion=${PYCHARM_VERSION} -Dpackaging=jar")
                runMaven(JDK_VERSION,"install:install-file -Dfile=pycharm.jar -DgroupId=com.jetbrains.pycharm -DartifactId=pycharm -Dversion=${PYCHARM_VERSION} -Dpackaging=jar")
                runMaven(JDK_VERSION,"install:install-file -Dfile=pycharm-pydev.jar -DgroupId=com.jetbrains.pycharm -DartifactId=pycharm-pydev -Dversion=${PYCHARM_VERSION} -Dpackaging=jar")
                runMaven(JDK_VERSION,"install:install-file -Dfile=resources_en.jar -DgroupId=com.jetbrains.pycharm -DartifactId=resources_en   -Dversion=${PYCHARM_VERSION} -Dpackaging=jar")
                runMaven(JDK_VERSION,"install:install-file -Dfile=util.jar -DgroupId=com.jetbrains.pycharm -DartifactId=util -Dversion=${PYCHARM_VERSION} -Dpackaging=jar")
                runMaven(JDK_VERSION,"install:install-file -Dfile=jps-model.jar -DgroupId=com.jetbrains.pycharm -DartifactId=jps-model -Dversion=${PYCHARM_VERSION} -Dpackaging=jar")
              }
              runMaven(JDK_VERSION,"clean install -Dskip.its=true -e")
            }
          }
         }
      }         
      post {
        always {
          sendAllNotificationQaResult()
        }
      }

    }
    stage('Promote') {
      steps {
        repoxPromoteBuild()
      }
      post {
        always {
          sendAllNotificationPromote()
        }
      }
    }
  }
}

def runITs(TEST,SQ_VERSION) {    
  withMaven(maven: MAVEN_TOOL) {
    mavenSetBuildVersion()        
    gitFetchSubmodules()
    dir("its/$TEST") {    
      runMavenOrch(JDK_VERSION,"verify -Dsonar.runtimeVersion=$SQ_VERSION")
    }
  }
}
