import difflib
import json

with open("data1.json", "r") as f1:
    data1 = json.load(f1)

with open("data2.json", "r") as f2:
    data2 = json.load(f2)

d = difflib.Differ()
diff = d.compare(data1["text"], data2["text"])
htdiff = difflib.HtmlDiff().make_file(data1["text"], data2["text"])

with open("diff.html", "w") as f:
    f.write(htdiff)

with open("diff.json", "w") as f:
    json.dump(list(diff), f)
