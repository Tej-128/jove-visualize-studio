# JoVE Visualize Studio

An invited-user pilot that accepts a plain numbered script and produces an MP4 review edit and a portable Shotcut timeline. Timestamps, video IDs, spreadsheets and a separately generated voiceover are not required inputs.

## Current workflow

1. Run the copied JoVE Visualizer matcher against the prepared private caption index.
2. Generate narration with the approved ElevenLabs profile. Independent steps run with bounded concurrency; existing narration is reused after script and profile checks.
3. Inspect suggested timestamp windows first, then suggested videos for unresolved steps, then the wider available library. Prepared timestamped storyboards support videos without captions.
4. Select multiple complementary shots when one script step contains several actions. A narration-duration mismatch does not discard suitable footage.
5. Recheck selected windows with denser samples. Unobservable metadata is recorded separately from absent actions and explicit scientific contradictions.
6. Render real source footage on the complete narration clock. Review labels identify uncertain visuals and substantial footage-speed changes. Explicitly rejected windows cannot become verified fallback matches.
7. Deliver MP4, editable timeline with portable media, source in/out records, audio adjustments and a review report. No placeholder-card renderer is used by the default editor.

## Validation and limits

Local automated checks cover candidate-first routing from a plain script, multi-shot selection, uncertainty handling, rejection of invented sources and timestamps, provider-failure fallback, concurrent request accounting, real video rendering, narration frame coverage, pitch preservation, portable assets, invitation and owner checks, and archive/cache validation.

Local tests do not establish scientific accuracy or a ten-minute cloud runtime. Those require a real plain-script trial, comparison against withheld reference selections, playback review, and timing measurements. A related but unverified candidate is explicitly a review candidate. If no usable footage exists, the app must report the omitted step rather than manufacture a match. A partial edit is not a complete scientific production.

The default visual stage has a 240-second budget, up to three concurrent checks, and bounded provider requests. It retains completed work and moves to rendering when review resources are exhausted. Matching and speech calls also have bounded timeouts. The target is five to ten minutes for a short script against an already prepared library; first-time uploads, indexing and provider outages are separate limitations.

## Cloud data and access

Deploy this new repository with entry point `studio/app.py` and Python 3.11. Existing repositories and applications are independent.

Store actual credentials only in Streamlit secure settings, using `.streamlit/secrets.example.toml` as the field reference. Company scripts, captions, videos, voiceovers, generated edits and reports belong in approved private Drive folders, never the public repository. Google sign-in requires a verified email on the explicit invitation list. The app's Google callback is its own `/oauth2callback` URL.

The service account needs read access to the input Library and write access to a separate output folder in a company Shared Drive. No domain-wide delegation or organization-administrator access is needed. No billing setup or purchases are part of this deployment.

The input Library contains the SHA-256-bound caption index, source media (or verified transfer parts), and optional prepared title/visual overview caches. `prepare_library.py` and `prepare_chunks.py` prepare these assets separately from script processing. The server never needs a user's Windows folder.

Generated narration, matching reports and completed selections are checkpointed in the private job folder. New owner/script checkpoints are authenticated before cloud restoration. Interrupted signed jobs can be restored after a host restart and rebuilt without reuploading the script. Older unsigned checkpoints are not automatically trusted for restoration.

## Editing policy

Narration stays at the approved natural profile. Selected footage is allocated across each step's narration duration; material video-speed adjustments are visibly flagged for review, never disguised as scientific verification. Source cuts and playback changes remain in `edit_plan.json`. Multiple shots may illustrate one step. The default editor uses cuts within an experiment and does not add a dissolve between every numbered step. Automatic experiment-boundary classification and eight-frame dissolves at verified experiment boundaries remain future work.

The editable package includes the prepared selected media and narration with relative paths. Source ranges and review limitations are retained, while the normalization and playback changes are baked into the packaged clip files. Original server media is not modified.

## Production work still required

Validate fresh scripts and withheld references; measure warm and cold cloud runtimes; calibrate visual decisions; test invited/uninvited users and multi-user load; validate host restart recovery; add per-shot review controls, retention/deletion policy, a durable independent worker service, experiment grouping, and account-level spending safeguards. Free Streamlit hosting is a pilot environment, not a guaranteed production service.