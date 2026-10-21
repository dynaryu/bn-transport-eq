#!/usr/bin/env bash
# Deploy the feed-only ShakeMap watcher to AWS Lambda, or update it.
#
# Safe to re-run: what exists is kept, and the function's code is replaced
# with a fresh zip. Run it again after changing a package's network, so the
# Lambda watches the new regions.
#
# One SNS topic carries both kinds of email: a new earthquake near a network
# (once per event), and the watcher having stopped.
#
# Usage (from WSL or Linux, logged in with `aws sso login`):
#   bash scripts/deploy-feedwatch.sh                       # deploy or update
#   bash scripts/deploy-feedwatch.sh you@example.com       # also subscribe this address
#
# Overrides (environment): AWS_REGION (ap-southeast-2), FEEDWATCH_BUCKET
# (bn-pilot-watch-<account>), FEEDWATCH_NAME (bn-pilot-feedwatch, used for the
# function, role, schedule and topic), PYTHON (uv's .venv, else the active python,
# e.g. after `conda activate dra`).
set -euo pipefail

EMAIL="${1:-}"
export AWS_REGION="${AWS_REGION:-ap-southeast-2}"
export AWS_DEFAULT_REGION="$AWS_REGION"
export AWS_PAGER=""
NAME="${FEEDWATCH_NAME:-bn-pilot-feedwatch}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# The project's interpreter: uv's .venv if there is one, else whatever python is
# active (e.g. the conda environment).
if [ -z "${PYTHON:-}" ]; then
  if [ -x "$ROOT/.venv/bin/python" ]; then PYTHON="$ROOT/.venv/bin/python"; else PYTHON="$(command -v python || command -v python3)"; fi
fi
ZIP="$ROOT/dist/feedwatch-lambda.zip"

step() { printf '\n== %s\n' "$*"; }

ACCOUNT=$(aws sts get-caller-identity --query Account --output text) \
  || { echo "Not logged in to AWS: run 'aws sso login' first." >&2; exit 1; }
BUCKET="${FEEDWATCH_BUCKET:-bn-pilot-watch-$ACCOUNT}"
echo "Account $ACCOUNT, region $AWS_REGION, bucket $BUCKET, function $NAME"

step "1. Bucket"
if aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  echo "exists"
else
  aws s3 mb "s3://$BUCKET"
fi

step "2. Email topic"
TOPIC=$(aws sns create-topic --name "$NAME" --query TopicArn --output text)
echo "$TOPIC"
if [ -n "$EMAIL" ]; then
  SUBSCRIBED=$(aws sns list-subscriptions-by-topic --topic-arn "$TOPIC" \
    --query "length(Subscriptions[?Endpoint=='$EMAIL'])" --output text)
  if [ "$SUBSCRIBED" = 0 ]; then
    aws sns subscribe --topic-arn "$TOPIC" --protocol email --notification-endpoint "$EMAIL" >/dev/null
    echo "subscribed $EMAIL: confirm the email AWS sends"
  else
    echo "$EMAIL already subscribed"
  fi
fi

step "3. Code: feedwatch.py plus each package's region"
(cd "$ROOT" && "$PYTHON" -m src.forward_pilot shakemap lambda-zip --output "$ZIP")

step "4. Role: logs, the bucket, the health metric, and the topic"
NEW_ROLE=0
if aws iam get-role --role-name "$NAME" >/dev/null 2>&1; then
  echo "exists"
else
  aws iam create-role --role-name "$NAME" --query Role.Arn --output text --assume-role-policy-document \
    '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
  NEW_ROLE=1
fi
aws iam attach-role-policy --role-name "$NAME" \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
aws iam put-role-policy --role-name "$NAME" --policy-name feedwatch --policy-document "{
  \"Version\": \"2012-10-17\", \"Statement\": [
  {\"Effect\": \"Allow\", \"Action\": \"s3:ListBucket\", \"Resource\": \"arn:aws:s3:::$BUCKET\"},
  {\"Effect\": \"Allow\", \"Action\": [\"s3:GetObject\", \"s3:PutObject\"], \"Resource\": \"arn:aws:s3:::$BUCKET/*\"},
  {\"Effect\": \"Allow\", \"Action\": \"sns:Publish\", \"Resource\": \"$TOPIC\"},
  {\"Effect\": \"Allow\", \"Action\": \"cloudwatch:PutMetricData\", \"Resource\": \"*\",
   \"Condition\": {\"StringEquals\": {\"cloudwatch:namespace\": \"BNPilot\"}}}]}"
if [ "$NEW_ROLE" = 1 ]; then
  echo "waiting for the new role to become usable"
  sleep 15
fi

step "5. Function (one poll at a time)"
if aws lambda get-function --function-name "$NAME" >/dev/null 2>&1; then
  aws lambda update-function-code --function-name "$NAME" --zip-file "fileb://$ZIP" \
    --query LastModified --output text
  aws lambda wait function-updated-v2 --function-name "$NAME"
  # Add the topic to the settings it already has; ones changed in the console
  # (e.g. FEEDWATCH_MIN_MAGNITUDE) are kept.
  CURRENT=$(aws lambda get-function-configuration --function-name "$NAME" \
    --query Environment.Variables --output json)
  if [ "$(echo "$CURRENT" | "$PYTHON" -c 'import json,sys; print((json.load(sys.stdin) or {}).get("FEEDWATCH_TOPIC_ARN", ""))')" != "$TOPIC" ]; then
    MERGED=$(echo "$CURRENT" | TOPIC="$TOPIC" "$PYTHON" -c \
      'import json,os,sys; v=json.load(sys.stdin) or {}; v["FEEDWATCH_TOPIC_ARN"]=os.environ["TOPIC"]; print(json.dumps({"Variables": v}))')
    aws lambda update-function-configuration --function-name "$NAME" --environment "$MERGED" \
      --query LastModified --output text
    aws lambda wait function-updated-v2 --function-name "$NAME"
  fi
else
  aws lambda create-function --function-name "$NAME" --runtime python3.13 \
    --handler feedwatch.lambda_handler --zip-file "fileb://$ZIP" \
    --role "arn:aws:iam::$ACCOUNT:role/$NAME" --memory-size 128 --timeout 120 \
    --environment "Variables={FEEDWATCH_BUCKET=$BUCKET,FEEDWATCH_TOPIC_ARN=$TOPIC}" --query FunctionArn --output text
  aws lambda wait function-active-v2 --function-name "$NAME"
fi
aws lambda put-function-concurrency --function-name "$NAME" --reserved-concurrent-executions 1 \
  --query ReservedConcurrentExecutions --output text

step "6. Schedule: every 2 minutes"
aws events put-rule --name "$NAME" --schedule-expression "rate(2 minutes)" --query RuleArn --output text
if aws lambda get-policy --function-name "$NAME" --query Policy --output text 2>/dev/null | grep -q '"Sid":"schedule"'; then
  echo "permission exists"
else
  aws lambda add-permission --function-name "$NAME" --statement-id schedule \
    --action lambda:InvokeFunction --principal events.amazonaws.com \
    --source-arn "arn:aws:events:$AWS_REGION:$ACCOUNT:rule/$NAME" >/dev/null
fi
aws events put-targets --rule "$NAME" \
  --targets "Id=1,Arn=arn:aws:lambda:$AWS_REGION:$ACCOUNT:function:$NAME" \
  --query FailedEntryCount --output text

step "7. Alert: no healthy poll for 10 minutes"
aws cloudwatch put-metric-alarm --alarm-name "$NAME-down" \
  --namespace BNPilot --metric-name WatcherHealthy --statistic Maximum --period 300 \
  --evaluation-periods 2 --threshold 1 --comparison-operator LessThanThreshold \
  --treat-missing-data breaching --alarm-actions "$TOPIC" --ok-actions "$TOPIC"
echo "alarm $NAME-down -> $TOPIC"

step "8. Test poll"
OUT=$(mktemp)
aws lambda invoke --function-name "$NAME" "$OUT" --query StatusCode --output text
cat "$OUT"; echo
rm -f "$OUT"

echo
echo "Done. New earthquakes near the networks are emailed to the topic's subscribers."
