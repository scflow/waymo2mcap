import urllib.request
import json

query = """
query {
  samples(dataset: "waymo-episodes-video", view: []) {
    ... on SampleItemStrConnection {
      edges {
        node {
          ... on Sample {
            id
            sample
          }
        }
      }
      total
    }
  }
}
"""

req = urllib.request.Request("http://127.0.0.1:5151/graphql", data=json.dumps({"query": query}).encode("utf-8"), headers={"Content-Type": "application/json"})
resp = urllib.request.urlopen(req)
data = json.loads(resp.read().decode("utf-8"))
print("GraphQL raw data:", data)
print("Total:", res.get("total"))
edges = res.get("edges", [])
print("Edges count:", len(edges))
for e in edges:
    s = json.loads(e["node"]["sample"])
    print("  id:", e["node"]["id"], "episode:", s.get("episode_id")[:36], "video:", s.get("filepath"))
