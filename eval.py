from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
A4VL_PATH = ROOT / "egoschema_results.json"
LVAGENT_PATH = ROOT / "results" / "egoschema_results.json"


def load_results(path: Path) -> list[dict]:
    with path.open("r") as fp:
        return json.load(fp)


def gold_answer(item: dict) -> str:
    return chr(ord("A") + int(item["correct_choice"]))


def final_answer(item: dict) -> str:
    return str(item.get("final_answer", ""))[:1]


def round_name(item: dict) -> str:
    if "third_samp" in item:
        return "third"
    if "second_samp" in item:
        return "second"
    if "final_answer" in item:
        return "first"
    return "unfinished"


def summarize(name: str, data: list[dict]) -> list[dict]:
    done = [item for item in data if "final_answer" in item]
    correct = sum(final_answer(item) == gold_answer(item) for item in done)

    print(f"\n{name}")
    print(f"Items: {len(data)}")
    print(f"Finished: {len(done)}")
    print(f"Correct: {correct}")
    print(f"Accuracy on finished: {correct / len(done):.4f}" if done else "Accuracy on finished: n/a")
    print(f"Accuracy on all items: {correct / len(data):.4f}" if data else "Accuracy on all items: n/a")

    for stage in ("first", "second", "third"):
        stage_items = [item for item in done if round_name(item) == stage]
        stage_correct = sum(final_answer(item) == gold_answer(item) for item in stage_items)
        acc = stage_correct / len(stage_items) if stage_items else 0.0
        print(f"{stage.title()} round finals: {stage_correct}/{len(stage_items)} ({acc:.4f})")

    first_vote_stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for item in done:
        answer_dict = item.get("first_round", {}).get("answer_dict", {})
        for agent_name, answer in answer_dict.items():
            first_vote_stats[agent_name][1] += 1
            first_vote_stats[agent_name][0] += str(answer)[:1] == gold_answer(item)

    if first_vote_stats:
        print("First-round model vote accuracy:")
        for agent_name in sorted(first_vote_stats):
            correct_votes, total_votes = first_vote_stats[agent_name]
            print(f"  {agent_name}: {correct_votes}/{total_votes} ({correct_votes / total_votes:.4f})")

    return done


def compare_aligned(a4vl_done: list[dict], lvagent_done: list[dict]) -> None:
    a4vl_by_video = {item["video_path"]: item for item in a4vl_done}
    lvagent_by_video = {item["video_path"]: item for item in lvagent_done}
    shared_videos = sorted(set(a4vl_by_video) & set(lvagent_by_video))

    a4vl_correct = 0
    lvagent_correct = 0
    a4vl_only = 0
    lvagent_only = 0
    both_correct = 0
    both_wrong = 0
    round_counter = Counter()
    first_vote_stats: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])

    for video_path in shared_videos:
        a4vl_item = a4vl_by_video[video_path]
        lvagent_item = lvagent_by_video[video_path]
        gold = gold_answer(a4vl_item)
        a4vl_ok = final_answer(a4vl_item) == gold
        lvagent_ok = final_answer(lvagent_item) == gold

        a4vl_correct += a4vl_ok
        lvagent_correct += lvagent_ok
        a4vl_only += a4vl_ok and not lvagent_ok
        lvagent_only += lvagent_ok and not a4vl_ok
        both_correct += a4vl_ok and lvagent_ok
        both_wrong += not a4vl_ok and not lvagent_ok
        round_counter[(round_name(a4vl_item), round_name(lvagent_item), a4vl_ok, lvagent_ok)] += 1

        a4vl_first = a4vl_item.get("first_round", {}).get("answer_dict", {})
        lvagent_first = lvagent_item.get("first_round", {}).get("answer_dict", {})
        for agent_name in sorted(set(a4vl_first) | set(lvagent_first)):
            if agent_name in a4vl_first:
                first_vote_stats[agent_name][1] += 1
                first_vote_stats[agent_name][0] += str(a4vl_first[agent_name])[:1] == gold
            if agent_name in lvagent_first:
                first_vote_stats[agent_name][3] += 1
                first_vote_stats[agent_name][2] += str(lvagent_first[agent_name])[:1] == gold

    print("\nAligned Comparison")
    print(f"Shared finished videos: {len(shared_videos)}")
    print(f"A4VL correct: {a4vl_correct}/{len(shared_videos)} ({a4vl_correct / len(shared_videos):.4f})")
    print(f"LVAgent correct: {lvagent_correct}/{len(shared_videos)} ({lvagent_correct / len(shared_videos):.4f})")
    print(f"Both correct: {both_correct}")
    print(f"Both wrong: {both_wrong}")
    print(f"A4VL only correct: {a4vl_only}")
    print(f"LVAgent only correct: {lvagent_only}")

    print("Aligned first-round model vote accuracy:")
    for agent_name in sorted(first_vote_stats):
        a4vl_hits, a4vl_total, lv_hits, lv_total = first_vote_stats[agent_name]
        a4vl_acc = a4vl_hits / a4vl_total if a4vl_total else 0.0
        lv_acc = lv_hits / lv_total if lv_total else 0.0
        print(
            f"  {agent_name}: "
            f"A4VL {a4vl_hits}/{a4vl_total} ({a4vl_acc:.4f}), "
            f"LVAgent {lv_hits}/{lv_total} ({lv_acc:.4f})"
        )

    print("Top round/correctness patterns:")
    for key, count in round_counter.most_common(12):
        print(f"  {key}: {count}")


def main() -> None:
    a4vl_done = summarize("A4VL", load_results(A4VL_PATH))
    lvagent_done = summarize("LVAgent", load_results(LVAGENT_PATH))
    compare_aligned(a4vl_done, lvagent_done)


if __name__ == "__main__":
    main()
