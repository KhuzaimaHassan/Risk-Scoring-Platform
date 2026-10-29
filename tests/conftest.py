import os

# These tests must never connect to the configured hosted database.
os.environ.update(ENVIRONMENT="test", DEBUG="false", MONITORING_ENABLED="false",
                  ADMIN_PASSWORD="TestOnly-LongPassword!", DB_HOST="127.0.0.1", DB_PORT="1")
