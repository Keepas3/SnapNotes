#!/bin/sh
# Newly created volumes are owned by root; hand /data to the app user, then run the command as that user.
set -e

if [ "$(id -u)" = "0" ]; then
    chown -R snapnotes:snapnotes "${SNAPNOTES_DATA_DIR:-/data}"
    exec setpriv --reuid=snapnotes --regid=snapnotes --init-groups "$@"
fi

exec "$@"
