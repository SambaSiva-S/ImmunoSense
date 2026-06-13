# Web v9 — photo preview thumbnail (closes the "I see nothing after attaching" gap)

After attaching a meal photo you now see a 44px thumbnail preview next to the
filename (from the compressed image, in-browser — no extra API call). Confirms
the right photo attached.

Pairs with backend immunosense_photo_fix.tar.gz (which fixes the 502 so the
upload actually reaches the bucket). The GET /v1/photo/{id} view endpoint is now
available for showing photos in History later (not yet wired into History — that
needs reflections to carry photo refs, a separate change).

## Run: npm install ; npm run dev
