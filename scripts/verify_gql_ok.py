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

samples_res = data["data"]["samples"]
edges = samples_res["edges"]
print("Successfully retrieved video samples from FiftyOne App!")
print("Number of Episode video samples:", len(edges))
for i, e in enumerate(edges, 1):
    s = e["node"]["sample"]
    if isinstance(s, str):
        s = json.loads(s)
    print(f"[{i}] id: {e['node']['id']} | episode: {s.get('episode_id', '')[:35]} | video: {s.get('filepath')}")
