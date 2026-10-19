#!/bin/bash
# Determined runs this once, inside the container, when a shell starts.
# Start: det shell start --config-file shell.yaml --context shell
# It makes `det shell open` land in the project directory instead of
# /run/determined/workdir.

grep -q 'mlde_wsp_RAI_WS/ar44ruca' "$HOME/.bashrc" 2>/dev/null || cat >> "$HOME/.bashrc" <<'EOF'
export LC_ALL=C.UTF-8
cd /pfss/mlde/workspaces/mlde_wsp_RAI_WS/ar44ruca 2>/dev/null || true
EOF

# Login shells (ssh) read .bash_profile/.profile, not .bashrc.
touch "$HOME/.bash_profile"
grep -q '\.bashrc' "$HOME/.bash_profile" || \
  echo '[ -f ~/.bashrc ] && . ~/.bashrc' >> "$HOME/.bash_profile"
