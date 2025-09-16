#!/bin/sh

echo "Generating .env from environment variables..."

cat <<EOF > /app/env/.env
DB_URL=${DBURL}
DB_USERNAME=${DBUSERNAME}
DB_PASSWORD=${DBPASSWORD}
EOF
