"""Generate seeds.train.txt (200) + seeds.eval.txt (20) — non-overlapping."""
import random

random.seed(42)
all_seeds = set()
while len(all_seeds) < 220:
    # 8-char uppercase alphanumeric, matches typical STS2 seed style
    s = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=10))
    all_seeds.add(s)
all_seeds = sorted(all_seeds)
train = all_seeds[:200]
eval_ = all_seeds[200:]

with open("seeds.train.txt", "w") as f:
    f.write("\n".join(train) + "\n")
with open("seeds.eval.txt", "w") as f:
    f.write("\n".join(eval_) + "\n")
print(f"seeds.train.txt: {len(train)} seeds (first 3: {train[:3]})")
print(f"seeds.eval.txt:  {len(eval_)} seeds (first 3: {eval_[:3]})")
