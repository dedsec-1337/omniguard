#!/usr/bin/env bash
# One-command start for school exhibitions / offline demos.
# Uses Fast profile (friendly to 4 GB VRAM cards like GTX 1650) and never
# tries the network after models are already cached.

set -e
cd "$(dirname "$0")"

if [ -d "venv" ]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
elif [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export HF_HUB_OFFLINE=1
export OMNIGUARD_PROFILE="${OMNIGUARD_PROFILE:-fast}"
# Optional: force a single light detector if you still hit VRAM pressure
# export OMNIGUARD_DETECTORS=gary

echo "OMNIGUARD exhibition mode"
echo "  profile = $OMNIGUARD_PROFILE"
echo "  offline = on"
echo "  open the URL printed below in your browser"
echo ""

python app.py
