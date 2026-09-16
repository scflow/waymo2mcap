import fiftyone as fo
import fiftyone.types as fot
import fiftyone.core.threed as fo3d

print("fiftyone types:")
for x in dir(fot):
    if not x.startswith("_"):
        print(" ", x)

print("\nfiftyone 3d:")
for x in dir(fo3d):
    if not x.startswith("_"):
        print(" ", x)
