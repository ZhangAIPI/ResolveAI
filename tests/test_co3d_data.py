"""CO3D adapter provenance and availability, using tiny software fixtures."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from resolveai.co3d_data import prepare, source_split
from resolveai.environment import Environment


class CO3DTests(unittest.TestCase):
    def test_official_task_groups_placeholder_masks_and_private_provenance(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name); source=root/"source"; output=root/"cases"
            category=source/"chair"; (category/"set_lists").mkdir(parents=True)
            (source/"source_manifest.json").write_text(json.dumps({"revision":"fixture","license":"fixture only"}))
            frames=[]
            for number in range(1,5):
                prefix="chair/private-sequence"
                image=prefix+f"/images/frame{number}.jpg"; mask=prefix+f"/masks/frame{number}.png"
                (source/image).parent.mkdir(parents=True,exist_ok=True); (source/mask).parent.mkdir(parents=True,exist_ok=True)
                Image.new("RGB",(16,16)).save(source/image)
                Image.new("L",(16,16),0 if number==3 else 255).save(source/mask)
                frames.append({"sequence_name":"private-sequence","frame_number":number,"frame_timestamp":number/30,
                               "image":{"path":image,"size":[16,16]},"mask":{"path":mask}})
            with gzip.open(category/"frame_annotations.jgz","wt") as stream:json.dump(frames,stream)
            rows=[["private-sequence",f["frame_number"],f["image"]["path"]] for f in frames]
            (category/"set_lists/set_lists_fewview_test.json").write_text(json.dumps({"train":rows[:2],"test":rows[2:]}))
            manifest=prepare(source,output,["chair"],families=1,views=3)
            self.assertEqual(manifest["cases"],4);self.assertEqual(manifest["original_images"],3)
            self.assertEqual(source_split(source,"chair"),{"private-sequence":"test"})
            cases=json.loads((output/"cases.json").read_text())
            env=Environment(cases[1],output)
            observation=env.observation()
            self.assertNotIn("private-sequence",json.dumps({k:v for k,v in observation.items() if k!="images"}))
            target=cases[1]["annotation"]["subclaims"][0]["minimal_evidence_sets"]["Supported"][0][0]["right"]["image_id"]
            row=next(r for r in cases[1]["evidence"] if r["id"]==target)
            query={"type":"request_photo","query":{k:row[k] for k in ("object","time","view")}}
            self.assertEqual(env.step(query)["status"],"provided")
            a,b=Environment(cases[2],output),Environment(cases[3],output)
            self.assertEqual(a.step(query),b.step(query))
            provenance=json.loads((output/"provenance.json").read_text())
            self.assertEqual([p["frame_number"] for p in provenance],[1,2,4])
            self.assertTrue(all(p["sequence"]=="private-sequence" for p in provenance))
            self.assertEqual(cases[0]["annotation"]["status"],"source-derived-unreviewed")
            (category/"set_lists/set_lists_fewview_train.json").write_text(json.dumps({"train":rows}))
            self.assertEqual(source_split(source,"chair"),{})
