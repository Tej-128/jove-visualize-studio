"""Build visual review windows from transcript hits and their adjacent passages."""

def attach_review_windows(db, candidates):
    for candidate in candidates:
        article = candidate["id"]
        start = candidate["start_ms"]
        bounds = [(start, candidate["end_ms"])]
        for comparator, order in (("<", "DESC"), (">", "ASC")):
            neighbor = db.execute(
                f"SELECT start_ms, end_ms FROM passages WHERE article_id=? "
                f"AND start_ms {comparator} ? ORDER BY start_ms {order} LIMIT 1",
                (article, start),
            ).fetchone()
            # Avoid jumping across discontinuities or long gaps in captions.
            if neighbor and neighbor[0] <= bounds[0][1] + 10000 and neighbor[1] >= start - 10000:
                bounds.append(neighbor)
        candidate["visual_review_ms"] = [min(a for a, _ in bounds), max(b for _, b in bounds)]

def adjacent_candidates(db, hits):
    """Return unseen flanking intervals in chunks small enough for visual inspection."""
    candidates = []
    seen = set()
    for hit in hits:
        start = round(hit["start"] * 1000)
        end = round(hit["end"] * 1000)
        item = {"id": hit["video_id"], "start_ms": start, "end_ms": end}
        attach_review_windows(db, [item])
        left, right = item["visual_review_ms"]
        for a, b in ((left, start), (end, right)):
            for begin in range(a, b, 45000):
                stop = min(b, begin + 45000)
                key = (item["id"], begin, stop)
                if stop - begin < 400 or key in seen:
                    continue
                seen.add(key)
                candidates.append({"video_id": item["id"], "start": begin/1000,
                                   "end": stop/1000, "caption": ""})
    return candidates
