dependencies = ['torch', 'torchaudio']

def bigvsan(pretrained=True):
    import torch
    from supervoice_vocoder.bigvsan import BigVSAN
    model = BigVSAN()
    model.eval()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url("https://shared.korshakov.com/models/bigvsan_10m_no_norm.pt")
        model.load_state_dict(checkpoint)
    return model
            