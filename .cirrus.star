load("github.com/SonarSource/cirrus-modules@v3", "load_features")
load("cirrus", "env", "fs", "yaml")
load(
    "github.com/SonarSource/cirrus-modules/cloud-native/helper.star@analysis/master",
    "merge_dict"
)

def private_conf(ctx):
    conf = fs.read("private/.cirrus.yml")

    features = load_features(ctx)
    features = yaml.dumps(features)
    return features + conf

# workaround for BUILD-4413 (build number on public CI)
def build_4413_workaround():
    return {
        'env': {
            'CI_BUILD_NUMBER': env.get("CIRRUS_PR", "1")
        },
    }

def public_conf(ctx):
    conf = fs.read(".cirrus-public.yml")
    features = load_features(ctx)
    features = yaml.dumps(features)
    return features + conf

def is_public_repo():
    return env.get("CIRRUS_REPO_FULL_NAME") == 'SonarSource/sonar-python' or env.get("CIRRUS_BRANCH").startswith("public-ci/")

def main(ctx):
    if is_public_repo():
        return {} 
    else:
        return private_conf(ctx)
