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

## Editor version 4 pilot update

Caption retrieval now restricts semantic, lexical and title candidates to videos that actually exist in the configured library **before** ranking. The copied original matcher is unchanged. A reusable script-independent visual action catalogue supplements captions, including videos without captions. Prepare it once from **Library readiness** using existing credits; observations are approximate search candidates and must still pass close visual checks for each new script. Private catalogue files are bound to the current source checksums and stored only in the new app's output folder.

Narration runs alongside matching. The renderer rejects insufficient footage that would create extreme slow motion. `quality.json` separates artifact generation, action coverage, timing and publication approval. Automatic list numbering and table order are retained for normal Word inputs. Reviewers can inspect selected clips by script step and reject a step's shots; a retry reuses saved matching, narration and unchanged rendered clips. No benchmark selection is an editor input. `evaluate_edit.py` compares generated plans against private withheld references after generation; overlap alone is not scientific approval.

Job requests use a SQLite queue with leases, heartbeats, duplicate-request prevention and source-build handover. Provider credentials stay in worker memory, not the queue. Free Community Cloud still shares its process and temporary disk with the UI; this does not provide a production uptime guarantee. `worker.py`, `Dockerfile` and `compose.yaml` prepare an optional separate worker deployment on a persistent shared volume. These files do not buy hosting or deploy an independent worker. Multi-host scheduling, measured load capacity and production infrastructure remain release work.

The 5–10 minute target applies to a fresh short script against a prepared library, and must be measured including fresh matching, narration, selection and rendering. Library ingestion is a separate one-time process. Reports identify reused stages so a rebuild cannot be passed off as a fresh run.

## Independent scientific audit (editor version 5)

The first version-4 fresh trial met the runtime target but independent playback review found unsuitable scientific substitutions. Version 5 adds an independent review of the exact final cuts, using source introductions, local source captions and final-cut frames without the selector's favourable explanation. It distinguishes live experiment footage from animation, rejects unrelated section boundaries, and separates directly visible actions from inferences. Source contradictions cannot be promoted to accepted matches. Honest illustrations remain labelled review candidates while the editor searches for demonstrated alternatives. This is an additional AI check, not a guarantee of scientific correctness; a version-5 cloud validation is still required.

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
