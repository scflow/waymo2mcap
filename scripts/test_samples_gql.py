import urllib.request
import json

req = urllib.request.Request("http://127.0.0.1:5151/graphql", data=json.dumps({
    "query": """
    query {
      samples(dataset: "waymo-episodes-video", view: []) {
        edges {
          node {
            id
            sample
          }
        }
        total
      }
    }
    """
}).encode("utf-8"), headers={"Content-Type": "application/json"})

resp = urllib.request.urlopen(req)
data = json.loads(resp.read().decode("utf-8"))
print("Samples raw response:", data)
print("Total samples:", samples_data.get("total"))
edges = samples_data.get("edges", [])
print("Edges count:", len(edges))
if edges:
    first_sample = json.loads(edges[0]["node"]["sample"])
    print("First sample keys:", list(first_sample.keys()))
    print("filepath:", first_sample.get("filepath"))
    print("episode_id:", first_sample.get("episode_id"))
