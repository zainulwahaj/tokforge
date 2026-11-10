#!/bin/bash
# Installs or updates the tokforge demo on the VPS. Run as root from a directory
# holding: tokforge (Linux x86_64 release binary), MANIFEST.json, the three
# <encoding>.tokenizer.json files, index.html, tokforge@.service, tokforge.caddy.
set -euo pipefail
cd "$(dirname "$0")"

declare -A PORTS=([gpt2]=18101 [cl100k_base]=18102 [o200k_base]=18103)

# Refuse model files that do not match the pinned manifest.
for enc in "${!PORTS[@]}"; do
  want=$(python3 -c "import json,sys; m=json.load(open('MANIFEST.json')); a=m['artifacts'] if 'artifacts' in m else m; a=a if isinstance(a,list) else list(a.values()); print(next(x['sha256'] for x in a if (x.get('name') or x.get('id'))==sys.argv[1]))" "$enc")
  have=$(sha256sum "$enc.tokenizer.json" | cut -d' ' -f1)
  [ "$want" = "$have" ] || { echo "$enc.tokenizer.json does not match MANIFEST.json"; exit 1; }
done

install -d -m 755 /opt/tokforge/bin /opt/tokforge/models /opt/tokforge/www /etc/tokforge
install -m 755 tokforge /opt/tokforge/bin/tokforge
for enc in "${!PORTS[@]}"; do
  install -m 644 "$enc.tokenizer.json" /opt/tokforge/models/
  echo "PORT=${PORTS[$enc]}" > "/etc/tokforge/$enc.env"
done
install -m 644 index.html /opt/tokforge/www/index.html
install -m 644 tokforge@.service /etc/systemd/system/tokforge@.service

systemctl daemon-reload
for enc in "${!PORTS[@]}"; do
  systemctl enable --quiet "tokforge@$enc"
  systemctl restart "tokforge@$enc"
done

# Caddy site, rolled back if the config does not validate.
site=/etc/caddy/sites/tokforge.caddy
[ -f "$site" ] && cp "$site" "$site.bak"
install -m 644 tokforge.caddy "$site"
if caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile >/dev/null 2>&1; then
  systemctl reload caddy
  rm -f "$site.bak"
else
  caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile 2>&1 | tail -5
  if [ -f "$site.bak" ]; then mv "$site.bak" "$site"; else rm -f "$site"; fi
  echo "Caddy config invalid, site change rolled back"; exit 1
fi

sleep 1
for enc in "${!PORTS[@]}"; do
  printf '%-12s %s\n' "$enc" "$(systemctl is-active "tokforge@$enc")"
done
