# Source on the Mac: reaches the 42 master through an SSH SOCKS tunnel via ias (no TUDA VPN needed).
SP=/private/tmp/claude-501/-Users-noahfarr-boonta-BB-8/6184d1c9-89eb-4f9a-9abd-ae67b9f103bc/scratchpad
export DET_MASTER=https://login01.ai.tu-darmstadt.de:8080 HTTPS_PROXY=socks5h://127.0.0.1:18888 PYTHONPATH=$SP/socks SCRATCH=$SP
pgrep -f "ssh -f -N -D 18888" >/dev/null || ssh -f -N -D 18888 -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 ias
R=/pfss/mlde/workspaces/mlde_wsp_RAI_WS/ar44ruca
sync42() { (cd ~/boonta-BB-8 && rsync -az -e "ssh -J ias -o BatchMode=yes" --exclude=.venv --exclude=outputs --exclude=sweeps --exclude=__pycache__ --exclude=.pytest_cache ./ 42:$R/bb8/ && { git rev-parse HEAD; git status --short; } | on42 "cat > $R/bb8/REVISION"); }
on42() { ssh -J ias -o BatchMode=yes 42 "$@"; }
