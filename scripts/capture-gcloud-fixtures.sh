#!/usr/bin/env bash
# Capture real gcloud output for fixtures and vendor-fact checks (TRD §17).
# READ-ONLY: every command goes through `run`, which refuses anything whose
# verb is not on the allow-list below. Nothing is created, changed or deleted.
#
#   scripts/capture-gcloud-fixtures.sh --project <id> [--region asia-southeast1]
#                                      [--bucket <name>] [--out fixtures-capture]
#                                      [--only <name>[,<name>...]]
#
# --only re-runs just the named captures (the names printed as "== name:").
#
# Output goes to a git-ignored folder (default: fixtures-capture/). It contains
# your project id, numbers and emails: trim and anonymise before anything is
# copied into tests/fakes/gcloud/. Never commit the folder.
set -u

PROJECT=""
REGION="asia-southeast1"
BUCKET=""
OUT="fixtures-capture"
ONLY=""

while [ $# -gt 0 ]; do
  case "$1" in
    --project) PROJECT="$2"; shift 2 ;;
    --region) REGION="$2"; shift 2 ;;
    --bucket) BUCKET="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --only) ONLY=",$2,"; shift 2 ;;
    -h|--help) sed -n 2,14p "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

if [ -z "$PROJECT" ]; then
  echo "usage: $0 --project <id> [--region R] [--bucket B] [--out DIR]" >&2
  exit 2
fi
BUCKET="${BUCKET:-ml-runner-$PROJECT}"
SA="ml-runner-vm@$PROJECT.iam.gserviceaccount.com"
mkdir -p "$OUT/stderr"

# Allow-list: a command is run only if it contains one of these read-only verbs
# and none of the mutating ones.
READ_VERBS=" version list describe describe-from-family get-iam-policy get-value cat "
MUTATING=" create delete add-iam-policy-binding remove-iam-policy-binding enable disable update set-iam-policy cp rsync mv rm "

run() { # run <name> <gcloud args...>
  local name="$1"; shift
  if [ -n "$ONLY" ]; then case "$ONLY" in *",$name,"*) ;; *) return 0 ;; esac; fi
  local ok=0 word
  for word in "$@"; do
    case "$MUTATING" in *" $word "*) echo "REFUSED (mutating): $*" >&2; return 1 ;; esac
    case "$READ_VERBS" in *" $word "*) ok=1 ;; esac
  done
  if [ "$ok" -ne 1 ]; then echo "REFUSED (not allow-listed): $*" >&2; return 1; fi
  echo "== $name: gcloud $*"
  gcloud "$@" >"$OUT/$name.out" 2>"$OUT/stderr/$name.txt"
  local code=$?
  echo "$code" >"$OUT/$name.exit"
  echo "   exit $code"
}

P="--project=$PROJECT"
J="--format=json"

run version                 version "$J"
run auth_list               auth list --filter=status:ACTIVE "$J"
run config_project          config get-value project
run projects_describe       projects describe "$PROJECT" "$J"
run services_enabled        services list --enabled "$P" "$J"
run bucket_describe         storage buckets describe "gs://$BUCKET" "$P" "$J"
run bucket_policy           storage buckets get-iam-policy "gs://$BUCKET" "$P" "$J"
run sa_describe             iam service-accounts describe "$SA" "$P" "$J"
run role_describe           iam roles describe mlRunnerSelfDelete "$P" "$J"
run project_policy          projects get-iam-policy "$PROJECT" "$J"
# accelerator-types reports `zone` as a URL (.../zones/<zone>), so the filter is not anchored.
run accelerator_types       compute accelerator-types list "$P" --filter="zone~$REGION-" "$J"
run region_describe         compute regions describe "$REGION" "$P" "$J"
run project_info            compute project-info describe "$P" "$J"
run network_default         compute networks describe default "$P" "$J"
run images_cuda             compute images list --project=deeplearning-platform-release --no-standard-images --filter="family~common-cu" "$J"
# No CPU family matched "common-cpu"/"base-cpu": list every DLVM family, and the debian-12 fallback (TRD §10.1).
run images_families         compute images list --project=deeplearning-platform-release --no-standard-images --format="json(family,name,description)"
run image_debian12          compute images describe-from-family debian-12 --project=debian-cloud "$J"
run machine_types_g2        compute machine-types list "$P" --filter="name=g2-standard-4 AND zone~$REGION-" "$J"

# Error samples (still read-only): stderr text for the classifier.
run err_bucket_missing      storage buckets describe "gs://mlr-no-such-bucket-$RANDOM$RANDOM" "$P" "$J"
run err_object_missing      storage objects describe "gs://$BUCKET/sources/mlr-no-such-object-$RANDOM$RANDOM.tar.gz" "$P" "$J"
run err_sa_missing          iam service-accounts describe "no-such-sa@$PROJECT.iam.gserviceaccount.com" "$P" "$J"
run err_instance_missing    compute instances describe mlr-no-such-instance --zone="$REGION-a" "$P" "$J"

echo
echo "Done. Output in $OUT/ (git-ignored). Failures above are expected for resources"
echo "that do not exist yet (bucket, service account, role, policies before init)."
echo "Paste back: $OUT/*.out and $OUT/stderr/*.txt, or tell me the folder is ready"
echo "if I may read it. Quota-exceeded / zone-exhausted stderr cannot be captured"
echo "read-only and stay provisional."
