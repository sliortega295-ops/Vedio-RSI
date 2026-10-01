# Phase 3 parent-side remote proxy authentication probe

Scope: the authorized A100 node, the user-provided institutional proxy endpoint
`114.111.19.82:80`, and a `HEAD` request for the fixed official AlexNet URL.
This was a connectivity/authentication test, **not** a weight download.

- A false credential first confirmed that the interactive path produced no
  terminal echo. The proxy returned HTTP CONNECT 407 (`curl` exit 56,
  origin `http=000`).
- The credential from the user's earlier example was then tested through the
  same path and also returned HTTP CONNECT 407. The value is intentionally
  omitted here; it did not authorize this A100 route.
- The proxy credential was supplied via an interactive standard-input read
  with local and remote terminal echo disabled. `curl` received an in-memory
  `--config -` stream, not a proxy URL or credential in its process arguments.
  No proxy setting, credential file, HTTP body, or model weight was written to
  the project or remote system. The command output contained only the 407/error
  status. The tool-call transcript may retain the input text, so this should
  **not** be described as secret-free or zero-exposure.
- Neither the local VPN nor a direct official-URL fallback was used. No GPU job
  was submitted.

Result: **BLOCKED_AUTH**. A working remote-side credential supplied through a
trusted channel, or an approved no-auth institution mirror URL, is still
needed before the exact AlexNet SHA can be staged and the original LPIPS judge
can run. The installed LPIPS package remains separate and unpromoted.
