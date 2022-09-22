from aws_cdk import (aws_rds as rds)

validKey = "my_valid_key"
encrypted = True
my_key = kms.Key(self, "Key", removal_policy=RemovalPolicy.DESTROY)
noneKey = None
not_encrypted = False

## DatabaseCluster
rds.DatabaseCluster(self, "encrypted1", storage_encrypted=True)
rds.DatabaseCluster(self, "encrypted1", storage_encrypted=encrypted)
rds.DatabaseCluster(self, "encrypted2", storage_encryption_key="my_key")
rds.DatabaseCluster(self, "encrypted3", storage_encryption_key=validKey)
rds.DatabaseCluster(self, "encrypted4", storage_encrypted=False, storage_encryption_key=my_key)
rds.DatabaseCluster(self, "encrypted5", storage_encrypted=True, storage_encryption_key=None)
rds.DatabaseCluster(self, "unencrypted") # NonCompliant
rds.DatabaseCluster(self, "unencrypted", storage_encrypted=False) # NonCompliant
rds.DatabaseCluster(self, "unencrypted", storage_encrypted=not_encrypted) # NonCompliant
rds.DatabaseCluster(self, "unencrypted", storage_encryption_key=None) # NonCompliant
rds.DatabaseCluster(self, "unencrypted", storage_encryption_key=noneKey) # NonCompliant
rds.DatabaseCluster(self, "unencrypted", storage_encrypted=False, storage_encryption_key=None) # NonCompliant

## DatabaseInstance
rds.DatabaseInstance(self, "encrypted1", storage_encrypted=True)
rds.DatabaseInstance(self, "encrypted1", storage_encrypted=encrypted)
rds.DatabaseInstance(self, "encrypted2", storage_encryption_key=my_key)
rds.DatabaseInstance(self, "encrypted3", storage_encrypted=False, storage_encryption_key=my_key)
rds.DatabaseInstance(self, "encrypted4", storage_encrypted=True, storage_encryption_key=None)
rds.DatabaseInstance(self, "unencrypted") # NonCompliant
rds.DatabaseInstance(self, "unencrypted", storage_encrypted=False ) # NonCompliant
rds.DatabaseInstance(self, "unencrypted", storage_encrypted=not_encrypted ) # NonCompliant
rds.DatabaseInstance(self, "unencrypted", storage_encryption_key=None) # NonCompliant
rds.DatabaseInstance(self, "unencrypted", storage_encryption_key=noneKey) # NonCompliant
rds.DatabaseInstance(self, "unencrypted", storage_encrypted=False, storage_encryption_key=None) # NonCompliant

## CfnDBCluster
rds.CfnDBCluster(self, "encrypted1", engine="any engine", storage_encrypted=True) # has to be always set for encryption, unlike DatabaseCluster
rds.CfnDBCluster(self, "encrypted2", engine="any engine", storage_encrypted=encrypted) # has to be always set for encryption, unlike DatabaseCluster
rds.CfnDBCluster(self, "unencrypted", engine="any engine") # NonCompliant
rds.CfnDBCluster(self, "unencrypted", engine="any engine", storage_encrypted=False) # NonCompliant
rds.CfnDBCluster(self, "unencrypted", engine="any engine", storage_encrypted=not_encrypted) # NonCompliant

## CfnDBInstance
# if aurora engine is used, no encryption can be done, so not reporting this case
auroraEngine = "aurora"
notAuroraEngine = "random engine"
rds.CfnDBInstance(self, "encrypted1", engine="mysql", storage_encrypted=True) # has to be always set for encryption, unlike DatabaseInstance
rds.CfnDBInstance(self, "encrypted2", engine="mysql", storage_encrypted=encrypted)
rds.CfnDBInstance(self, "cant-encrypt", engine="aurora")
rds.CfnDBInstance(self, "cant-encrypt", engine="aurora", storage_encrypted=True)
rds.CfnDBInstance(self, "cant-encrypt", engine="aurora", storage_encrypted=encrypted)
rds.CfnDBInstance(self, "cant-encrypt", engine="aurora", storage_encrypted=False)
rds.CfnDBInstance(self, "cant-encrypt", engine="aurora engine")
rds.CfnDBInstance(self, "cant-encrypt", engine=auroraEngine)
rds.CfnDBInstance(self, "cant-encrypt", engine="AuRoRa")
rds.CfnDBInstance(self, "unencrypted", engine="mysql") # NonCompliant
rds.CfnDBInstance(self, "unencrypted", engine="not aurora") # NonCompliant
rds.CfnDBInstance(self, "unencrypted", engine=notAuroraEngine) # NonCompliant
rds.CfnDBInstance(self, "unencrypted", engine="mysql", storage_encrypted=False) # NonCompliant
rds.CfnDBInstance(self, "unencrypted", engine="mysql", storage_encrypted=not_encrypted) # NonCompliant


## Not reported : not aws_cdk lib
from fakelib import (aws_rds as rds_fake)

rds_fake.DatabaseCluster(self, "unencrypted")
rds_fake.DatabaseInstance(self, "unencrypted", storage_encrypted=False )
rds_fake.CfnDBCluster(self, "unencrypted", engine="aurora")
rds_fake.CfnDBInstance(self, "unencrypted", engine="mysql")
