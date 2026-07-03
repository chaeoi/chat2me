#!/bin/sh
set -eu

. /entrypoint-lib.sh

init_config

exec "$@"
