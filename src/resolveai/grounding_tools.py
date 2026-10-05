"""Frozen visual proposal tools. Similarity/detection is never identity truth."""
from pathlib import Path


class FrozenGrounding:
    def __init__(self, root, device="cuda:0"):
        self.root, self.device = Path(root), device
        self._models = {}

    def available(self, kind):
        return any((self.root / kind).glob("*.safetensors"))

    def _load(self, kind):
        if kind not in self._models:
            from transformers import (AutoImageProcessor, AutoModel, AutoProcessor,
                                      AutoModelForZeroShotObjectDetection, CLIPModel)
            path = self.root / kind
            if kind == "text_image":
                processor, model = AutoProcessor.from_pretrained(path), AutoModelForZeroShotObjectDetection.from_pretrained(path, disable_custom_kernels=True)
            elif kind == "image_image":
                processor, model = AutoImageProcessor.from_pretrained(path), AutoModel.from_pretrained(path)
            else:
                processor, model = AutoProcessor.from_pretrained(path), CLIPModel.from_pretrained(path)
            self._models[kind] = processor, model.to(self.device).eval()
        return self._models[kind]

    def text_image(self, image, text):
        import torch
        processor, model = self._load("text_image")
        inputs = processor(images=image, text=text.strip().rstrip(".") + ".", return_tensors="pt").to(self.device)
        with torch.inference_mode():
            output = model(**inputs)
        result = processor.post_process_grounded_object_detection(output, inputs.input_ids,
            threshold=.25, text_threshold=.25, target_sizes=[image.size[::-1]])[0]
        labels = result.get("text_labels", result.get("labels"))
        return [{"bbox": box.tolist(), "score": float(score), "text": str(label)}
                for box, score, label in zip(result["boxes"], result["scores"], labels)]

    def image_text(self, image, candidates):
        import torch
        processor, model = self._load("image_text")
        inputs = processor(images=image, text=candidates, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            output = model(**inputs)
            similarities = output.image_embeds @ output.text_embeds.T
        return sorted([{"text": text, "similarity": float(score)}
                       for text, score in zip(candidates, similarities[0])],
                      key=lambda row: row["similarity"], reverse=True)

    def image_image(self, query, target, top_k=3):
        """Coarse DINOv2 dense feature proposals, not a calibrated identity detector."""
        import torch
        from PIL import Image
        processor, model = self._load("image_image")
        def features(image, side):
            image = image.resize((side, side), Image.Resampling.BILINEAR)
            inputs = processor(images=image, do_resize=False, do_center_crop=False, return_tensors="pt").to(self.device)
            return model(**inputs).last_hidden_state[0, 1:]
        with torch.inference_mode():
            q = torch.nn.functional.normalize(features(query, 224).mean(0), dim=0)
            patches = torch.nn.functional.normalize(features(target, 448), dim=-1)
            scores = (patches @ q).reshape(32, 32).cpu()
        # Connected high-similarity patches form candidate regions.
        threshold = float(torch.quantile(scores.flatten(), .9))
        pending = {(y, x) for y in range(32) for x in range(32) if float(scores[y, x]) >= threshold}
        groups = []
        while pending:
            stack = [min(pending)]
            pending.remove(stack[0])
            component = []
            while stack:
                y, x = stack.pop()
                component.append((y, x))
                for neighbor in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
                    if neighbor in pending:
                        pending.remove(neighbor)
                        stack.append(neighbor)
            ys, xs = zip(*component)
            groups.append({"bbox": [min(xs)*target.width/32, min(ys)*target.height/32,
                                    (max(xs)+1)*target.width/32, (max(ys)+1)*target.height/32],
                           "similarity": max(float(scores[y,x]) for y,x in component)})
        return sorted(groups, key=lambda row: row["similarity"], reverse=True)[:top_k]
