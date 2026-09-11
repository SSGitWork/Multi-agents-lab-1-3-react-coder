#!/bin/bash
echo "Starting secure environment setup (Generating config.yaml)..."

CONFIG_DIR="$HOME/.continue"
CONFIG_FILE="$CONFIG_DIR/config.yaml"
HELICONE_API_KEY="sk-helicone-a2uncjy-xwleneq-qbj4wji-lfphjfi"
OPENROUTER_API_KEY="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"

echo "Installing Node.js dependencies..."
npm install

GIT_EMAIL=$(git config user.email)
echo "Extracting username from $GIT_EMAIL..."

if [[ "$GIT_EMAIL" == *users.noreply.github.com ]]; then
    EMAIL_PREFIX=$(echo "$GIT_EMAIL" | sed 's/@.*//')
    FINAL_USERNAME=$(echo "$EMAIL_PREFIX" | sed -E 's/^[0-9]+\+//')
    if [ -z "$FINAL_USERNAME" ]; then
        FINAL_USERNAME=$(git config github.user)
    fi
else
    FINAL_USERNAME=$(echo "$GIT_EMAIL" | sed 's/@.*//')
fi

echo "Writing configuration file to $CONFIG_FILE..."
mkdir -p "$CONFIG_DIR" || true

cat > "$CONFIG_FILE" <<- EOF
name: Local Config
version: 1.0.0
schema: v1
models:
  - name: OpenRouter-via-Helicone
    provider: openai
    model: openrouter/auto
    apiBase: https://openrouter.helicone.ai/api/v1
    apiKey: '$OPENROUTER_API_KEY'

  - name: GPT-4-1-mini
    provider: openai
    model: openai/gpt-4.1-mini
    apiBase: https://openrouter.helicone.ai/api/v1
    apiKey: '$OPENROUTER_API_KEY'

  - name: GPT-5-nano
    provider: openai
    model: openai/gpt-5-nano
    apiBase: https://openrouter.helicone.ai/api/v1
    apiKey: '$OPENROUTER_API_KEY'

  - name: gpt-5.1-codex-mini
    provider: openai
    model: openai/gpt-5.1-codex-mini
    apiBase: https://openrouter.helicone.ai/api/v1
    apiKey: '$OPENROUTER_API_KEY'
roles:
  - chat
  - edit
  - apply
requestOptions:
  headers:
    Helicone-Auth: "Bearer $HELICONE_API_KEY"
    Helicone-User-Id: "$FINAL_USERNAME"
    Helicone-Course: "Multi-Agent"
EOF

if [ -f "$CONFIG_FILE" ]; then
    echo "Configuration file successfully written and ready for Continue AI."
else
    echo "FATAL ERROR: Failed to write configuration file."
fi

echo ""
echo "✅ Setup complete! Run: npm run dev"
echo "Please Reload Window to load the Continue AI configuration."
