import difflib
import json

with open("data1.json", "r") as f1:
    data1 = json.load(f1)

with open("data2.json", "r") as f2:
    data2 = json.load(f2)
htmldiff = difflib.HtmlDiff()
diff = htmldiff.make_file(
    data1["text"], data2["text"], context=False, numlines=len(data1["text"])
)

with open("diff.html", "w") as f:
    f.writelines(diff)
