import urllib.request
import json

req = urllib.request.Request("http://127.0.0.1:5151/graphql", data=json.dumps({
    "query": """
    query {
      datasets(search: "") {
        edges {
          node {
            name
            mediaType
          }
        }
      }
    }
    """
}).encode("utf-8"), headers={"Content-Type": "application/json"})

resp = urllib.request.urlopen(req)
data = json.loads(resp.read().decode("utf-8"))
print("GraphQL raw response:", data)
