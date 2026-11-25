#!/usr/bin/env bash
# One-command start for school exhibitions / 4 GB VRAM laptops (GTX 1650 class).
# Offline + Fast profile. Models stay in RAM and borrow the GPU one at a time.

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
export OMNIGUARD_SWAP="${OMNIGUARD_SWAP:-on}"
export OMNIGUARD_VRAM_GB="${OMNIGUARD_VRAM_GB:-3.2}"
# If you still hit VRAM pressure, uncomment the next line (one light detector only):
# export OMNIGUARD_DETECTORS=gary

echo "OMNIGUARD exhibition mode (4 GB VRAM safe)"
echo "  profile  = $OMNIGUARD_PROFILE"
echo "  swap     = $OMNIGUARD_SWAP"
echo "  budget   = ${OMNIGUARD_VRAM_GB} GB"
echo "  offline  = on"
echo "  open the URL printed below in your browser"
echo ""

python app.py
