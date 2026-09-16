import urllib.request
import json

def test_query(q):
    req = urllib.request.Request(
        "http://127.0.0.1:5151/graphql",
        data=json.dumps({"query": q}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))

for ds_name in ["waymo-multimodal-frames", "waymo-episodes-overview"]:
    q = f'''
    query {{
      dataset(name: "{ds_name}") {{
        name
        mediaType
      }}
      samples(dataset: "{ds_name}", view: [], first: 1) {{
        ... on SampleItemStrConnection {{
          total
          edges {{
            node {{
              ... on Sample {{
                id
                sample
              }}
            }}
          }}
        }}
      }}
    }}
    '''
    res = test_query(q)
    err = res.get("errors")
    if err:
        print(f"FAILED {ds_name}: {err}")
    else:
        edges = res.get("data", {}).get("samples", {}).get("edges", [])
        if edges:
            s0 = edges[0]["node"]["sample"]
            print(f"SUCCESS {ds_name}:")
            print(f"  Episode ID: {s0.get('episode_id')}")
            print(f"  Speed: {s0.get('speed_mps')} m/s | 3D Objs: {s0.get('num_objects_3d')}")
            if "ground_truth_2d" in s0:
                print(f"  2D Boxes: {len(s0['ground_truth_2d'].get('detections', []))} detections")
            if "objects_per_frame" in s0:
                print(f"  Objects/frame: {s0.get('objects_per_frame')} | Avg speed: {s0.get('avg_speed_kmh'):.1f} km/h")
