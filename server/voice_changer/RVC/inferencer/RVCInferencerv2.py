import torch
import json
from safetensors import safe_open
from const import EnumInferenceTypes
from voice_changer.common.deviceManager.DeviceManager import DeviceManager
from voice_changer.RVC.inferencer.Inferencer import Inferencer
from .rvc_models.infer_pack.models import SynthesizerTrnMs768NSFsid
from voice_changer.common.SafetensorsUtils import load_model


class RVCInferencerv2(Inferencer):
    def load_model(self, file: str):
        device_manager = DeviceManager.get_instance()
        dev = device_manager.device
        is_half = device_manager.use_fp16()
        self.set_props(EnumInferenceTypes.pyTorchRVCv2, file)

        # Keep torch.load for backward compatibility, but discourage the use of this loading method
        if file.endswith('.safetensors'):
            with safe_open(file, 'pt', device=str(dev) if dev.type == 'cuda' else 'cpu') as cpt:
                config = json.loads(cpt.metadata()['config'])
                model = SynthesizerTrnMs768NSFsid(*config, is_half=is_half).to(dev)
                load_model(model, cpt, strict=False)
        else:
            cpt = torch.load(file, map_location=dev if dev.type == 'cuda' else 'cpu')
            model = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=is_half).to(dev)
            model.load_state_dict(cpt["weight"], strict=False)

        model.eval().remove_weight_norm()

        if is_half:
            model = model.half()

        self.model = model
        return self

    def infer(
        self,
        feats: torch.Tensor,
        pitch_length: torch.Tensor,
        pitch: torch.Tensor,
        pitchf: torch.Tensor,
        sid: torch.Tensor,
        skip_head: torch.Tensor | None,
        return_length: torch.Tensor | None,
    ) -> torch.Tensor:
        if pitch is None or pitchf is None:
            raise RuntimeError("[Voice Changer] Pitch or Pitchf is not found.")

        res = self.model.infer(feats, pitch_length, pitch, pitchf, sid, skip_head=skip_head, return_length=return_length)
        res = res[0][0, 0].float()
        return torch.clip(res, -1.0, 1.0, out=res)
